/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "ast.h"
#include "astgen.h"
#include "cabi.h"
#include "codegen.h"
#include "init_analysis.h"
#include "jam_llvm.h"
#include "jir_codegen.h"
#include "jir_verify.h"
#include "lexer.h"
#include "mangling.h"
#include "module_resolver.h"
#include "parser.h"
#include "symbol_table.h"
#include "target.h"
#include <filesystem>

// Terminal progress reporting via OSC 9;4. Supported by ConEmu,
// Windows Terminal, iTerm2, WezTerm, Ghostty, etc. — renders as a
// taskbar / tab indicator (no inline animation that fights with
// diagnostic output). Unsupported terminals silently drop the
// escape sequence.
//
// States: 0=remove, 1=normal+percent, 2=error, 3=indeterminate,
// 4=warning. We use 3 during compile, 2 on error, 0 on success.
class ProgressGuard {
	bool active = false;

  public:
	ProgressGuard(bool enabled) {
		if (!enabled) return;
		if (!isatty(STDERR_FILENO)) return;
		active = true;
		std::cerr << "\033]9;4;3;\033\\" << std::flush;
	}
	~ProgressGuard() { clear(); }

	void error() {
		if (active) {
			std::cerr << "\033]9;4;2;\033\\" << std::flush;
			active = false;
		}
	}
	// Stop without flagging error — same exit as success.
	void stop() { clear(); }

  private:
	void clear() {
		if (active) {
			std::cerr << "\033]9;4;0;\033\\" << std::flush;
			active = false;
		}
	}
};

static int compileAndRun(const std::string &filename,
                         const std::string &outputName, bool runFlag,
                         bool emitIR, bool testMode, JamOptLevel optLevel,
                         JamLTO lto, JamStrip strip,
                         const std::vector<std::string> &linkLibs) {
	ProgressGuard progress(!testMode);

	std::ifstream file(filename);
	if (!file.is_open()) {
		progress.error();
		std::cerr << "Could not open file: " << filename << std::endl;
		return 1;
	}

	std::stringstream buffer;
	buffer << file.rdbuf();
	std::string source = buffer.str();

	JamCodegenContext codegenCtx("jam_module");
	codegenCtx.setCurrentFile(filename);
	Lexer lexer(source);
	std::vector<Token> tokens = lexer.scanTokens();

	std::vector<std::unique_ptr<StructDeclAST>> sharedAnonStructs;
	std::vector<std::unique_ptr<EnumDeclAST>> sharedAnonEnums;

	Parser parser(tokens, codegenCtx.getTypePool(), codegenCtx.getStringPool(),
	              codegenCtx.getNodeStore(), &codegenCtx.diagnostics(),
	              filename);
	parser.sharedAnonStructs = &sharedAnonStructs;
	parser.sharedAnonEnums = &sharedAnonEnums;
	std::unique_ptr<ModuleAST> module;
	try {
		module = parser.parse();
	} catch (const ParserAbort &) {
		// diagnostic already pushed
		progress.error();
		codegenCtx.diagnostics().emit(std::cerr);
		return 1;
	}
	std::filesystem::path sourcePath(filename);
	std::string baseDir = sourcePath.parent_path().string();
	if (baseDir.empty()) { baseDir = "."; }

	ModuleResolver resolver(baseDir, codegenCtx.getTypePool(),
	                        codegenCtx.getStringPool(),
	                        codegenCtx.getNodeStore());
	resolver.setSharedAnonRegistries(&sharedAnonStructs, &sharedAnonEnums);
	SymbolTable symbolTable;

	symbolTable.registerBuiltinSymbol("test", "assert");
	for (auto &import : module->Imports) {
		if (import->Path == "std" || import->Path == "test") { continue; }

		ModuleAST *importedModule = resolver.getOrLoadModule(import->Path);
		if (!importedModule) {
			std::cerr << "Error: Failed to load module: " << import->Path
			          << std::endl;
			return 1;
		}

		symbolTable.registerModule(import->Path, importedModule);
	}

	for (auto &destImport : module->DestructuringImports) {
		ModuleAST *importedModule = resolver.getOrLoadModule(destImport->Path);
		if (!importedModule) {
			std::cerr << "Error: Failed to load module: " << destImport->Path
			          << std::endl;
			return 1;
		}

		symbolTable.registerModule(destImport->Path, importedModule);
		for (const auto &name : destImport->Names) {
			if (!symbolTable.hasSymbol(destImport->Path, name)) {
				// Throw so the outer catch prefixes the importing
				// file's path on the error message.
				throw std::runtime_error("symbol `" + name +
				                         "` is not exported from module `" +
				                         destImport->Path + "`");
			}
			symbolTable.registerBinding(name, destImport->Path, name);
		}
	}

	codegenCtx.setAnonStructs(&sharedAnonStructs);
	codegenCtx.setAnonEnums(&sharedAnonEnums);
	// `publicOnly` is set when iterating an imported module: only `pub`
	// items get registered, so non-pub items can't leak into the
	// importing module via bare-name lookup. Main-module decls always
	// register since there's no outer consumer.
	auto declareStructs = [&](ModuleAST *m, bool publicOnly) {
		for (auto &s : m->Structs) {
			if (publicOnly && !s->isPub) continue;
			JamTypeRef structType = JamLLVMStructCreateNamed(
			    codegenCtx.getContext(), s->Name.c_str());
			codegenCtx.registerStruct(s->Name, structType, s->Fields);
		}
	};
	auto declareUnions = [&](ModuleAST *m, bool publicOnly) {
		for (auto &u : m->Unions) {
			if (publicOnly && !u->isPub) continue;
			JamTypeRef unionType = JamLLVMStructCreateNamed(
			    codegenCtx.getContext(), u->Name.c_str());
			codegenCtx.registerUnion(u->Name, unionType, u->Fields);
		}
	};
	auto fillStructBodies = [&](ModuleAST *m, bool publicOnly) {
		for (auto &s : m->Structs) {
			if (publicOnly && !s->isPub) continue;
			std::vector<JamTypeRef> fieldTypes;
			fieldTypes.reserve(s->Fields.size());
			for (auto &f : s->Fields) {
				fieldTypes.push_back(codegenCtx.getLLVMType(f.second));
			}
			const auto *info = codegenCtx.getStruct(s->Name);
			JamLLVMStructSetBody(info->type, fieldTypes.data(),
			                     static_cast<unsigned>(fieldTypes.size()),
			                     false);
		}
	};
	// Lay out a union as `{ alignedField, [paddingBytes x i8] }`. The
	// alignedField is the field with the largest alignment requirement;
	// padding makes up the difference between that field's size and the
	// largest field's size, so the union ends up with the right size and
	// alignment for any field's stored value.
	auto fillUnionBodies = [&](ModuleAST *m, bool publicOnly) {
		for (auto &u : m->Unions) {
			if (publicOnly && !u->isPub) continue;
			if (u->Fields.empty()) {
				throw std::runtime_error("Union `" + u->Name +
				                         "` must have at least one field");
			}
			uint64_t maxSize = 0, maxAlign = 1;
			size_t alignFieldIdx = 0;
			for (size_t i = 0; i < u->Fields.size(); i++) {
				uint64_t sz = codegenCtx.typeSize(u->Fields[i].second);
				uint64_t al = codegenCtx.typeAlign(u->Fields[i].second);
				if (sz > maxSize) maxSize = sz;
				if (al > maxAlign) {
					maxAlign = al;
					alignFieldIdx = i;
				}
			}
			// Round size up to a multiple of alignment so writes through
			// the most-aligned field don't run off the end.
			uint64_t allocSize = (maxSize + maxAlign - 1) / maxAlign * maxAlign;
			JamTypeRef alignedTy =
			    codegenCtx.getLLVMType(u->Fields[alignFieldIdx].second);
			uint64_t alignedSz =
			    codegenCtx.typeSize(u->Fields[alignFieldIdx].second);
			uint64_t paddingBytes =
			    allocSize > alignedSz ? allocSize - alignedSz : 0;

			std::vector<JamTypeRef> bodyTypes;
			bodyTypes.push_back(alignedTy);
			if (paddingBytes > 0) {
				bodyTypes.push_back(
				    JamLLVMArrayType(codegenCtx.getInt8Type(),
				                     static_cast<unsigned>(paddingBytes)));
			}
			const auto *info = codegenCtx.getUnion(u->Name);
			JamLLVMStructSetBody(info->type, bodyTypes.data(),
			                     static_cast<unsigned>(bodyTypes.size()),
			                     false);
		}
	};
	auto declareEnums = [&](ModuleAST *m, bool publicOnly) {
		for (auto &e : m->Enums) {
			if (publicOnly && !e->isPub) continue;
			std::vector<JamCodegenContext::EnumVariantInfo> variants;
			variants.reserve(e->Variants.size());
			for (auto &v : e->Variants) {
				JamCodegenContext::EnumVariantInfo info;
				info.name = v.Name;
				info.payloadTypes = v.PayloadTypes;
				info.discriminant = v.Discriminant;
				variants.push_back(std::move(info));
			}
			codegenCtx.registerEnum(e->Name, std::move(variants));
		}
	};
	// For payloaded enums, lay out as `{i8 tag, alignDriver,
	// [extraBytes x i8]}` where `alignDriver` is the smallest scalar
	// type whose alignment matches the strictest variant's alignment
	// (i8 / i16 / i32 / i64 for align 1 / 2 / 4 / 8 respectively).
	// LLVM gives the resulting struct the alignment of `alignDriver`,
	// which propagates to allocas and stores — without that we get
	// align-1 enum slots even for u64-payload variants, which forces
	// LLVM to emit unaligned memory ops.
	//
	// `extraBytes` makes up the difference between the largest variant's
	// payload size and the alignDriver scalar's size, rounded up to
	// maxAlign so the trailing slot in an array of enums is correctly
	// aligned.
	auto fillEnumBodies = [&](ModuleAST *m, bool publicOnly) {
		for (auto &e : m->Enums) {
			if (publicOnly && !e->isPub) continue;
			const auto *info = codegenCtx.getEnum(e->Name);
			if (!info || !info->hasPayloadVariant) continue;

			uint64_t maxSize = 0, maxAlign = 1;
			for (const auto &v : info->variants) {
				uint64_t off = 0, varAlign = 1;
				for (TypeIdx t : v.payloadTypes) {
					uint64_t s = codegenCtx.typeSize(t);
					uint64_t a = codegenCtx.typeAlign(t);
					off = (off + a - 1) / a * a;  // align this field
					off += s;
					if (a > varAlign) varAlign = a;
				}
				if (varAlign > 1) {
					off = (off + varAlign - 1) / varAlign * varAlign;
				}
				if (off > maxSize) maxSize = off;
				if (varAlign > maxAlign) maxAlign = varAlign;
			}

			// Pick a scalar to drive struct alignment.
			JamTypeRef alignDriver;
			uint64_t alignDriverSize;
			switch (maxAlign) {
			case 1:
				alignDriver = codegenCtx.getInt8Type();
				alignDriverSize = 1;
				break;
			case 2:
				alignDriver = codegenCtx.getInt16Type();
				alignDriverSize = 2;
				break;
			case 4:
				alignDriver = codegenCtx.getInt32Type();
				alignDriverSize = 4;
				break;
			case 8:
				alignDriver = codegenCtx.getInt64Type();
				alignDriverSize = 8;
				break;
			default:
				throw std::runtime_error(
				    "Enum `" + e->Name +
				    "` requires alignment > 8, which is not yet "
				    "supported");
			}

			// Round payload size up to maxAlign so a contiguous array
			// of enums tiles correctly.
			uint64_t paddedSize =
			    (maxSize + maxAlign - 1) / maxAlign * maxAlign;
			uint64_t extraBytes = (paddedSize > alignDriverSize)
			                          ? paddedSize - alignDriverSize
			                          : 0;

			std::vector<JamTypeRef> bodyTypes;
			bodyTypes.push_back(codegenCtx.getInt8Type());  // tag
			bodyTypes.push_back(alignDriver);
			if (extraBytes > 0) {
				bodyTypes.push_back(
				    JamLLVMArrayType(codegenCtx.getInt8Type(),
				                     static_cast<unsigned>(extraBytes)));
			}

			JamLLVMStructSetBody(info->type, bodyTypes.data(),
			                     static_cast<unsigned>(bodyTypes.size()),
			                     false);
			codegenCtx.setEnumLLVMType(e->Name, info->type, maxSize, maxAlign,
			                           true);
		}
	};
	// Enums that need a named struct type (i.e. those with payload
	// variants) get their LLVM type created here, in declareEnums, so
	// that fillEnumBodies can set the body in a second pass.
	auto declareEnumLLVMTypes = [&](ModuleAST *m) {
		for (auto &e : m->Enums) {
			const auto *info = codegenCtx.getEnum(e->Name);
			if (!info || !info->hasPayloadVariant) continue;
			JamTypeRef ty = JamLLVMStructCreateNamed(codegenCtx.getContext(),
			                                         e->Name.c_str());
			codegenCtx.setEnumLLVMType(e->Name, ty, 0, 1, true);
		}
	};
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		if (path == "std") continue;
		declareStructs(importedModule.get(), /*publicOnly=*/true);
		declareUnions(importedModule.get(), /*publicOnly=*/true);
		declareEnums(importedModule.get(), /*publicOnly=*/true);
		declareEnumLLVMTypes(importedModule.get());
	}
	declareStructs(module.get(), /*publicOnly=*/false);
	declareUnions(module.get(), /*publicOnly=*/false);
	declareEnums(module.get(), /*publicOnly=*/false);
	declareEnumLLVMTypes(module.get());
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		if (path == "std") continue;
		fillStructBodies(importedModule.get(), /*publicOnly=*/true);
		fillUnionBodies(importedModule.get(), /*publicOnly=*/true);
		fillEnumBodies(importedModule.get(), /*publicOnly=*/true);
	}
	fillStructBodies(module.get(), /*publicOnly=*/false);
	fillUnionBodies(module.get(), /*publicOnly=*/false);
	fillEnumBodies(module.get(), /*publicOnly=*/false);

	// Register module-scope `const NAME[: T]? = expr;` bindings. These
	// are inlined at use sites (see AstTag::Variable in ast.cpp), so we
	// only need to teach the codegen context about them — no LLVM
	// globals get emitted.
	//
	// A const whose RHS is a type-call expression (e.g.
	// `const BoxI32 = Box(i32);`) is a *type alias*. We detect this
	// at registration time by walking the InitExpr AST: if the
	// expression is a Call whose callee is a registered generic with
	// return type `type`, we evaluate each arg as a type and bind
	// the result via `registerTypeAlias`. The parser stays grammar-
	// only; the type-vs-value decision lives at semantic time.
	const NodeStore &ns = codegenCtx.getNodeStore();

	// Look up a function by source-level name across the main module
	// and any imported pub fns. Functions aren't yet in the codegen
	// context's registry at this point (that happens in pass 1d).
	auto lookupGenericFn = [&](const std::string &name) -> const FunctionAST * {
		for (auto &fn : module->Functions) {
			if (fn->Name == name) return fn.get();
		}
		for (const auto &kv : resolver.getLoadedModules()) {
			if (kv.first == "std") continue;
			for (auto &fn : kv.second->Functions) {
				if (fn->isPub && fn->Name == name) return fn.get();
			}
		}
		return nullptr;
	};

	std::function<TypeIdx(NodeIdx)> resolveExprAsType =
	    [&](NodeIdx exprIdx) -> TypeIdx {
		const AstNode &n = ns.get(exprIdx);
		if (n.tag == AstTag::Variable) {
			const std::string &name =
			    codegenCtx.getStringPool().get(static_cast<StringIdx>(n.lhs));
			// Builtin scalar names.
			if (name == "u8") return BuiltinType::U8;
			if (name == "i8") return BuiltinType::I8;
			if (name == "u16") return BuiltinType::U16;
			if (name == "i16") return BuiltinType::I16;
			if (name == "u32") return BuiltinType::U32;
			if (name == "i32") return BuiltinType::I32;
			if (name == "u64") return BuiltinType::U64;
			if (name == "i64") return BuiltinType::I64;
			if (name == "f32") return BuiltinType::F32;
			if (name == "f64") return BuiltinType::F64;
			if (name == "bool" || name == "u1") return BuiltinType::Bool;
			if (name == "type") return BuiltinType::Type;
			if (name == "noreturn") return BuiltinType::NoReturn;
			// A bare identifier in a type-resolution context names
			// a Named user type (struct / enum / union). The
			// downstream type lookup at use-site decides whether the
			// reference resolves; from here we just produce the
			// Named TypeIdx.
			TypeIdx named = codegenCtx.getTypePool().internNamed(
			    codegenCtx.getStringPool().intern(name));
			return named;
		}
		if (n.tag == AstTag::Call) {
			// Indirect-call form (`expr.method()`) is never a type
			// alias. Direct-call form has the callee name in `lhs`.
			if ((n.flags & 1) != 0) return kNoType;
			const std::string &calleeName =
			    codegenCtx.getStringPool().get(static_cast<StringIdx>(n.lhs));
			const FunctionAST *fn = lookupGenericFn(calleeName);
			if (fn == nullptr) return kNoType;
			if (!fn->isGeneric()) return kNoType;
			if (fn->ReturnType != BuiltinType::Type) return kNoType;
			ExtraIdx argsExtra = static_cast<ExtraIdx>(n.rhs);
			uint32_t argCount = ns.getExtra(argsExtra);
			std::vector<TypeIdx> argTypes;
			argTypes.reserve(argCount);
			for (uint32_t i = 0; i < argCount; i++) {
				NodeIdx argIdx =
				    static_cast<NodeIdx>(ns.getExtra(argsExtra + 1 + i));
				TypeIdx argTy = resolveExprAsType(argIdx);
				if (argTy == kNoType) return kNoType;
				argTypes.push_back(argTy);
			}
			return codegenCtx.getTypePool().internGenericCall(
			    codegenCtx.getStringPool().intern(calleeName),
			    std::move(argTypes));
		}
		return kNoType;
	};

	auto registerConsts = [&](ModuleAST *m) {
		for (auto &c : m->Consts) {
			if (c->AliasedType != kNoType) {
				codegenCtx.registerTypeAlias(c->Name, c->AliasedType);
				continue;
			}
			if (c->InitExpr != kNoNode) {
				TypeIdx maybeAlias = resolveExprAsType(c->InitExpr);
				if (maybeAlias != kNoType) {
					const TypeKey &k = codegenCtx.getTypePool().get(maybeAlias);
					// Only bind as alias when the resolved type is
					// an actual user-visible category — generic
					// instantiations, struct/enum/union names. A bare
					// builtin (i32) or a Named lookup that doesn't
					// resolve fall through to value-const behavior.
					if (k.kind == TypeKind::GenericCall) {
						c->AliasedType = maybeAlias;
						codegenCtx.registerTypeAlias(c->Name, c->AliasedType);
						continue;
					}
				}
			}
			codegenCtx.registerModuleConst(c->Name, c->InitExpr,
			                               c->DeclaredType);
		}
	};
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		if (path == "std") continue;
		registerConsts(importedModule.get());
	}
	registerConsts(module.get());

	// Two-pass codegen: declare every function's prototype first, then
	// emit bodies. Without this, calling a function defined later in the
	// file (or another module) would fail with "Unknown function". The
	// two-pass shape lets source read naturally top-down (main on top,
	// helpers below) without a manual "forward declarations" section.

	// Pass 1a: prototypes for pub functions in imported modules.
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		if (path == "std") continue;
		for (auto &func : importedModule->Functions) {
			if (func->isPub && !func->isGeneric()) {
				JirFunction jfn = astgenMetadata(*func, codegenCtx);
				jfn.name = mangledFunctionName(*func, codegenCtx.getTypePool(),
				                               codegenCtx.getStringPool());
				jirDeclarePrototype(jfn, codegenCtx);
			}
			// Generic functions are still registered so call sites can
			// look them up for instantiation, but no LLVM is emitted
			// until each instantiation is processed.
			if (func->isPub) {
				codegenCtx.registerFunctionAST(func->Name, func.get());
			}
		}
	}

	for (auto &import : module->Imports) {
		if (import->Path == "std" || import->Path == "test") continue;
		const std::string &handle = import->Name;
		ModuleAST *importedModule = resolver.getOrLoadModule(import->Path);
		if (!importedModule) continue;
		codegenCtx.registerImportHandle(handle, import->Path);
		auto aliasNamed = [&](const std::string &bare) {
			TypeIdx target = codegenCtx.getTypePool().internNamed(
			    codegenCtx.getStringPool().intern(bare));
			codegenCtx.registerTypeAlias(handle + "." + bare, target);
		};
		// Track pub vs non-pub items per handle. Pub items get the
		// qualified registration; non-pub get recorded so a later
		// lookup miss produces the precise "not exported" diagnostic.
		for (auto &func : importedModule->Functions) {
			if (func->isPub) {
				codegenCtx.registerFunctionAST(handle + "." + func->Name,
				                               func.get());
			} else {
				codegenCtx.registerPrivateName(handle, func->Name);
			}
		}
		for (auto &s : importedModule->Structs) {
			if (s->isPub) aliasNamed(s->Name);
			else codegenCtx.registerPrivateName(handle, s->Name);
		}
		for (auto &e : importedModule->Enums) {
			if (e->isPub) aliasNamed(e->Name);
			else codegenCtx.registerPrivateName(handle, e->Name);
		}
		for (auto &u : importedModule->Unions) {
			if (u->isPub) aliasNamed(u->Name);
			else codegenCtx.registerPrivateName(handle, u->Name);
		}
	}

	// Pass 1b: prototypes for the main module's functions (we still skip
	// test funcs in non-test mode and user `main` in test mode).
	std::vector<std::string> testFunctionNames;
	std::vector<JirFunction> jirFunctions;
	for (auto &function : module->Functions) {
		if (function->isTest && !testMode) continue;
		if (!function->isTest && testMode && function->Name == "main") {
			continue;
		}
		if (function->isTest) {
			testFunctionNames.push_back("__test_" + function->Name);
		}
		// skip prototype + body emission for generic
		// functions. They get registered (so call sites can find them)
		// but no LLVM is emitted until an instantiation in supplies
		// concrete type arguments.
		// register by source-level name so call codegen can recover
		// parameter modes for callsite ABI decisions. Generic functions
		// also need to be in the registry — call sites consult it to
		// drive instantiation. The JIR astgen for bodies happens later
		// (after pass 1c struct methods are also registered), so any
		// astgen-time callee lookup can see every function in the
		// module.
		//
		// Test functions get the `__test_` prefix in their registry key
		// to avoid colliding with a regular function of the same source
		// name — Jam allows `fn add_u8(...)` and `tfn add_u8()` to coexist,
		// and the bare-name lookup in astgenCall must resolve to the
		// regular function, not its similarly-named test.
		const std::string regName =
		    function->isTest ? "__test_" + function->Name : function->Name;
		codegenCtx.registerFunctionAST(regName, function.get());
		// Non-generic main-module functions get their LLVM prototype
		// emitted by `jirDeclarePrototype` in pass 1d, alongside their
		// JirFunction. Generics emit nothing here — each instantiation
		// declares its own prototype in `instantiateStructExpr`.
	}

	// Methods declared inside struct bodies (`fn name(self: ..., ...)`).
	// v1 supports `drop` only; non-drop methods are accepted by the parser
	// so the surface stays forward-compatible, but rejected here so we
	// don't commit to method mangling yet. See docs/STRUCT_METHODS.md.
	auto resolveStructName = [&](TypeIdx ty) -> std::string {
		const auto &key = codegenCtx.getTypePool().get(ty);
		if (key.kind != TypeKind::Struct && key.kind != TypeKind::Named) {
			return "";
		}
		return codegenCtx.getStringPool().get(static_cast<StringIdx>(key.a));
	};
	for (auto &s : module->Structs) {
		for (auto &m : s->Methods) {
			// Two privileged method names on top-level structs:
			//   `drop`    — no-arg-cleanup; must take `self: mut Self`.
			//   `default` — opt-in default constructor; takes no
			//               parameters, returns `Self`. Used by anything
			//               that expects a default value (struct-literal
			//               omitted fields, generic type parameters that
			//               require a default, future `var x: T;`).
			// Anything else is rejected — there is no general user-
			// defined static-method or instance-method support on top-
			// level structs in v1.
			if (m->Name == "default") {
				if (!m->Args.empty()) {
					std::cerr << filename
					          << ": error: method `default` on struct `"
					          << s->Name << "` must take no parameters\n";
					return 1;
				}
				std::string retStruct = resolveStructName(m->ReturnType);
				if (retStruct != s->Name) {
					std::cerr
					    << filename << ": error: method `default` on struct `"
					    << s->Name << "` must return `Self` (got `" << retStruct
					    << "`)\n";
					return 1;
				}
			} else if (m->Name == "drop") {
				if (m->Args.empty() || m->Args[0].Name != "self") {
					std::cerr << filename << ": error: method `" << m->Name
					          << "` on struct `" << s->Name
					          << "` must take `self` as its first parameter\n";
					return 1;
				}
				std::string selfStruct = resolveStructName(m->Args[0].Type);
				if (selfStruct != s->Name) {
					std::cerr << filename << ": error: method `" << m->Name
					          << "` on struct `" << s->Name
					          << "` has self type `" << selfStruct
					          << "`; expected `" << s->Name << "`\n";
					return 1;
				}
			} else {
				std::cerr << filename
				          << ": error: only `drop` and `default` "
				             "methods are allowed on top-level "
				             "structs (saw `"
				          << s->Name << "." << m->Name << "`)\n";
				return 1;
			}
			{
				JirFunction jfn = astgenMetadata(*m, codegenCtx);
				jfn.name = mangledFunctionName(*m, codegenCtx.getTypePool(),
				                               codegenCtx.getStringPool());
				jirDeclarePrototype(jfn, codegenCtx);
			}
			codegenCtx.registerFunctionAST(s->Name + "." + m->Name, m.get());
		}
	}

	// Build the drop registry up front so AstGen can read it via
	// `JamCodegenContext::getDropRegistry()` to track which `var`
	// bindings need their drop fn called at scope exit.
	jam::drops::DropRegistry dropRegistry = jam::drops::buildDropRegistry(
	    *module, codegenCtx.getTypePool(), codegenCtx.getStringPool());
	codegenCtx.setDropRegistry(&dropRegistry);

	// Pass 1d: every function (free fns + struct methods + imported
	// pub fns) is registered. Run AstGen on each so call sites inside
	// their bodies can resolve every callee — including
	// `Counter.default`-style static-method calls that the parser
	// doesn't know about until pass 1c.
	//
	// All non-generic functions get their LLVM prototype from
	// `jirDeclarePrototype`. The classifier handles sret, by-pointer,
	// noreturn, zext, and callconv uniformly, so call sites and
	// definitions never disagree. Main-module free fns are declared
	// here in pass 1d; struct methods (pass 1c) and imported pub fns
	// (pass 1a) declare earlier so their bodies — also emitted via
	// JIR — can reference each other across pass boundaries.
	// Per-decl recovery: if astgen on one function throws
	// AstGenAnalysisFail, the diagnostic was already pushed to
	// codegenCtx.diagnostics(). We keep going so the user sees every
	// error in one pass instead of having to fix-rebuild-fix.
	for (auto &function : module->Functions) {
		if (function->isTest && !testMode) continue;
		if (!function->isTest && testMode && function->Name == "main") {
			continue;
		}
		if (function->isGeneric()) continue;
		try {
			JirFunction jfn = astgenFunction(*function, codegenCtx);
			jfn.name = mangledFunctionName(*function, codegenCtx.getTypePool(),
			                               codegenCtx.getStringPool());
			jirDeclarePrototype(jfn, codegenCtx);
			jirFunctions.push_back(std::move(jfn));
		} catch (const AstGenAnalysisFail &) {
			// diagnostic already pushed
		}
	}
	for (auto &s : module->Structs) {
		for (auto &m : s->Methods) {
			try {
				JirFunction jfn = astgenFunction(*m, codegenCtx);
				jfn.name = mangledFunctionName(*m, codegenCtx.getTypePool(),
				                               codegenCtx.getStringPool());
				jirFunctions.push_back(std::move(jfn));
			} catch (const AstGenAnalysisFail &) {
				// diagnostic already pushed
			}
		}
	}
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		if (path == "std") continue;
		for (auto &func : importedModule->Functions) {
			if (func->isPub && !func->isGeneric()) {
				try {
					JirFunction jfn = astgenFunction(*func, codegenCtx);
					jfn.name =
					    mangledFunctionName(*func, codegenCtx.getTypePool(),
					                        codegenCtx.getStringPool());
					jirFunctions.push_back(std::move(jfn));
				} catch (const AstGenAnalysisFail &) {
					// diagnostic already pushed
				}
			}
		}
	}

	// Astgen accumulated every per-decl diagnostic onto the
	// codegen context's global Diagnostics collector. If any
	// declaration failed, emit the whole batch now (sorted by
	// file:line) and bail before downstream passes — the verifier
	// and codegen will trip on the JirFunctions that *did* succeed
	// when their callees are missing prototypes.
	if (codegenCtx.diagnostics().hasErrors()) {
		progress.error();
		codegenCtx.diagnostics().emit(std::cerr);
		return 1;
	}

	// Verify each JirFunction's structural invariants before codegen
	// runs. Catches malformed dispatch, missing terminators,
	// out-of-bounds refs, and use-before-def across blocks — failures
	// that would otherwise show up as either silent miscompiles or
	// LLVM-verifier crashes much later. Diagnostics are aborts: the
	// function shouldn't reach jir_codegen if it's malformed.
	for (const JirFunction &jfn : jirFunctions) {
		auto diags = verifyJirFunction(
		    jfn, &codegenCtx.getTypePool(), &codegenCtx.getStringPool(),
		    +[](void *c, TypeIdx t) -> TypeIdx {
			    auto *cc = static_cast<JamCodegenContext *>(c);
			    const TypeKey &k = cc->getTypePool().get(t);
			    if (k.kind == TypeKind::GenericCall) {
				    TypeIdx r = cc->resolveGenericCall(t);
				    if (r != kNoType) return r;
			    }
			    return t;
		    },
		    &codegenCtx);
		for (auto &d : diags) {
			// jir_verify leaves file empty so the caller can stamp
			// the unit's currentFile here.
			if (d.loc.file.empty()) d.loc.file = filename;
			codegenCtx.diagnostics().push(std::move(d));
		}
	}
	if (codegenCtx.diagnostics().hasErrors()) {
		progress.error();
		codegenCtx.diagnostics().emit(std::cerr);
		return 1;
	}

	// Definite-init + mode-aware
	// callsite analysis runs after every prototype is in scope but
	// before any body is codegen'd. The drop registry was built
	// earlier (before pass 1d) so the JIR astgen could read it.
	{
		jam::init_analysis::FunctionRegistry fnRegistry;
		for (auto &fn : module->Functions) { fnRegistry[fn->Name] = fn.get(); }
		for (auto &s : module->Structs) {
			for (auto &m : s->Methods) {
				fnRegistry[s->Name + "." + m->Name] = m.get();
			}
		}
		for (const auto &kv : resolver.getLoadedModules()) {
			if (kv.first == "std") continue;
			for (auto &fn : kv.second->Functions) {
				if (fn->isPub) { fnRegistry[fn->Name] = fn.get(); }
			}
		}

		auto runAnalysis = [&](FunctionAST *function) {
			if (function->isExtern) return;
			auto diags = jam::init_analysis::analyze(
			    *function, codegenCtx.getNodeStore(),
			    codegenCtx.getStringPool(), tokens, &fnRegistry, &dropRegistry,
			    &codegenCtx.getTypePool());
			// Funnel each init-analysis diagnostic into the unified
			// `jam::Diagnostics` channel so they share the same
			// formatting / ordering as astgen errors.
			for (auto &d : diags) {
				jam::SrcLoc loc{filename, d.line};
				codegenCtx.diagnostics().error(loc, d.message);
			}
		};
		// Run init / drop analysis on every non-generic main-module
		// function and struct method. The pipeline registered all of
		// them in pass 1c / 1d, so this single sweep covers everything
		// emitted in pass 2b.
		for (auto &function : module->Functions) {
			if (function->isTest && !testMode) continue;
			if (!function->isTest && testMode && function->Name == "main")
				continue;
			if (function->isGeneric()) continue;
			runAnalysis(function.get());
		}
		for (auto &s : module->Structs) {
			for (auto &m : s->Methods) runAnalysis(m.get());
		}
		if (codegenCtx.diagnostics().hasErrors()) {
			progress.error();
			codegenCtx.diagnostics().emit(std::cerr);
			return 1;
		}

		// Pass 2b: emit each JirFunction's LLVM body. Every prototype
		// (main-module fns, struct methods, imported pub fns) was
		// declared in pass 1a–1c so jir_codegen Call lookups resolve.
		for (const JirFunction &jfn : jirFunctions) {
			jirDefineBody(jfn, codegenCtx);
		}
	}

	// In test mode, generate a main() that calls all test functions
	if (testMode) {
		if (testFunctionNames.empty()) {
			// No tests in this file: nothing to link or run.
			return 0;
		}

		JamTypeRef mainRetType = codegenCtx.getInt32Type();
		JamTypeRef mainFT = JamLLVMFunctionType(mainRetType, nullptr, 0, false);
		JamFunctionRef mainFunc =
		    JamLLVMAddFunction(codegenCtx.getModule(), "main", mainFT);
		JamLLVMApplyDefaultFnAttrs(mainFunc, /*isExtern=*/false);
		JamLLVMSetLinkage((JamValueRef)mainFunc, JAM_LINKAGE_EXTERNAL);
		JamLLVMSetFunctionCallConv(mainFunc, JAM_CALLCONV_C);

		JamBasicBlockRef entryBB = JamLLVMAppendBasicBlock(mainFunc, "entry");
		JamLLVMPositionBuilderAtEnd(codegenCtx.getBuilder(), entryBB);

		// Declare printf for test output
		JamFunctionRef printfFunc =
		    JamLLVMGetFunction(codegenCtx.getModule(), "printf");
		if (!printfFunc) {
			JamTypeRef i8PtrType =
			    JamLLVMPointerType(codegenCtx.getInt8Type(), 0);
			JamTypeRef printfParamTypes[1] = {i8PtrType};
			JamTypeRef printfType = JamLLVMFunctionType(
			    codegenCtx.getInt32Type(), printfParamTypes, 1, true);
			printfFunc = JamLLVMAddFunction(codegenCtx.getModule(), "printf",
			                                printfType);
			JamLLVMApplyDefaultFnAttrs(printfFunc, /*isExtern=*/true);
		}

		for (const auto &name : testFunctionNames) {
			JamFunctionRef testFunc =
			    JamLLVMGetFunction(codegenCtx.getModule(), name.c_str());
			if (testFunc) {
				// Print test name (strip __test_ prefix for display)
				std::string displayName = name.substr(7);  // remove "__test_"
				std::string msg = "testing " + displayName + "... ";
				JamValueRef msgStr = JamLLVMBuildGlobalStringPtr(
				    codegenCtx.getBuilder(), msg.c_str(), "test_msg");
				JamValueRef printArgs[1] = {msgStr};
				JamLLVMBuildCall(codegenCtx.getBuilder(), printfFunc, printArgs,
				                 1, "");

				// Call test function
				JamLLVMBuildCall(codegenCtx.getBuilder(), testFunc, nullptr, 0,
				                 "");

				// Print pass (if assert fails, exit(1) is called before
				// reaching here)
				JamValueRef passStr = JamLLVMBuildGlobalStringPtr(
				    codegenCtx.getBuilder(), "ok\n", "test_pass");
				JamValueRef passArgs[1] = {passStr};
				JamLLVMBuildCall(codegenCtx.getBuilder(), printfFunc, passArgs,
				                 1, "");
			}
		}

		// Print summary
		std::string summary =
		    std::to_string(testFunctionNames.size()) + " test(s) passed\n";
		JamValueRef summaryStr = JamLLVMBuildGlobalStringPtr(
		    codegenCtx.getBuilder(), summary.c_str(), "test_summary");
		JamValueRef summaryArgs[1] = {summaryStr};
		JamLLVMBuildCall(codegenCtx.getBuilder(), printfFunc, summaryArgs, 1,
		                 "");

		JamLLVMBuildRet(codegenCtx.getBuilder(),
		                JamLLVMConstInt(mainRetType, 0, false));
		JamLLVMVerifyFunction(mainFunc);
	}

	// Optionally print LLVM IR
	if (emitIR) {
		char *irStr = JamLLVMPrintModuleToString(codegenCtx.getModule());
		std::cout << irStr;
		JamLLVMDisposeMessage(irStr);
	}

	// Get target triple
	char *tripleStr = JamLLVMGetDefaultTargetTriple();
	JamLLVMSetTargetTriple(codegenCtx.getModule(), tripleStr);

	// must_pic is true on targets that need PIC (Windows DLLs, GNU
	// libc on Linux, …) and `pie` is true on PIE-by-default platforms
	// (Darwin). Without PIC, system `ld` on Linux x86_64 rejects
	// R_X86_64_32 absolute relocations when linking a PIE binary.
	const bool pic = []() {
		jam::Target t = jam::Target::getHostTarget();
		return t.requiresPIE() || t.requiresPIC();
	}();
	JamTargetMachineRef tm = JamLLVMCreateTargetMachine(tripleStr, "generic",
	                                                    "", pic, optLevel, lto);
	JamLLVMDisposeMessage(tripleStr);

	if (!tm) {
		std::cerr << "Failed to create target machine" << std::endl;
		return 1;
	}

	JamLLVMSetDataLayout(codegenCtx.getModule(), tm);

	// Emit object file (or LLVM bitcode when LTO is on; the linker re-runs
	// the optimization pipeline against the bitcode plus any LTO-compatible
	// static libs and produces the final binary).
	const std::string intermediate =
	    (lto == JAM_LTO_OFF) ? (outputName + ".o") : (outputName + ".bc");
	char *emitError = nullptr;
	bool success = JamLLVMEmitObjectFile(codegenCtx.getModule(), tm,
	                                     intermediate.c_str(), &emitError);

	JamLLVMDisposeTargetMachine(tm);

	if (!success) {
		std::cerr << "Failed to emit object file: "
		          << (emitError ? emitError : "unknown error") << std::endl;
		if (emitError) JamLLVMDisposeMessage(emitError);
		return 1;
	}

	// Link to create executable using system compiler. Append any -l flags
	// the user passed (`-lncurses`, `-l ncurses`, `--library ncurses`) so
	// extern fns from system libraries resolve. When LTO is on, hand the
	// bitcode to clang with `-flto=` so its driver picks the right linker
	// plugin (lld for ELF, ld64 for Mach-O, both with LTO support).
	std::string linkCmd = "clang " + intermediate + " -o " + outputName;
	if (lto == JAM_LTO_THIN) linkCmd += " -flto=thin";
	else if (lto == JAM_LTO_FAT) linkCmd += " -flto=full";
	for (const auto &lib : linkLibs) { linkCmd += " -l" + lib; }

	jam::Target host = jam::Target::getHostTarget();

	// libm linking. LLVM's `frem` on floating-point operands lowers
	// to a `fmod` libcall, so any `%` on a float needs the math
	// runtime visible to the linker. Rules per OS/libc:
	//   * macOS: libSystem bundles math; never pass `-lm`.
	//   * Windows: MSVCRT bundles math; never pass `-lm`.
	//   * Linux/FreeBSD + glibc: math lives in libm.so; pass `-lm`.
	//   * Linux/FreeBSD + musl: math is bundled into libc.a; passing
	//     `-lm` would fail on installs without a stub libm.a.
	// `linkLibc` is hard-coded to true today since Jam always links
	// libc; it becomes a real toggle when a freestanding compile mode
	// lands.
	const bool linkLibc = true;
	if (linkLibc &&
	    (host.os == jam::OS::Linux || host.os == jam::OS::FreeBSD) &&
	    host.abi != jam::ABI::Musl) {
		linkCmd += " -lm";
	}

	// Strip unreferenced functions/data at link time. Pairs with
	// FunctionSections / DataSections set on the TargetMachine, which split
	// each symbol into its own section so the linker can GC them
	// individually. Mach-O uses -dead_strip; ELF (Linux/FreeBSD) uses
	// --gc-sections. Skipped in debug to keep link fast.
	if (optLevel != JAM_OPT_NONE) {
		switch (host.os) {
		case jam::OS::MacOS:
			linkCmd += " -Wl,-dead_strip";
			break;
		case jam::OS::Linux:
		case jam::OS::FreeBSD:
			linkCmd += " -Wl,--gc-sections";
			break;
		case jam::OS::Windows:
		case jam::OS::Unknown:
			// PE/COFF link.exe uses /OPT:REF; lld-link the same. clang
			// driver translates --gc-sections appropriately when targeting
			// COFF, but the safer minimum is to not pass it from here.
			break;
		}
	}

	// Symbol/debug-info stripping at link time. Pure linker-flag plumbing.
	// Mach-O: -Wl,-S strips DWARF symbols; -Wl,-x removes local (non-global)
	// syms. ELF: --strip-debug for debug only, -s (a.k.a. --strip-all) for
	// everything. Symbols mode stacks debug+locals; rustc semantics.
	if (strip != JAM_STRIP_NONE) {
		switch (host.os) {
		case jam::OS::MacOS:
			linkCmd += " -Wl,-S";
			if (strip == JAM_STRIP_SYMBOLS) linkCmd += " -Wl,-x";
			break;
		case jam::OS::Linux:
		case jam::OS::FreeBSD:
			linkCmd +=
			    (strip == JAM_STRIP_SYMBOLS) ? " -Wl,-s" : " -Wl,--strip-debug";
			break;
		case jam::OS::Windows:
		case jam::OS::Unknown:
			break;
		}
	}
	int linkResult = system(linkCmd.c_str());
	if (linkResult != 0) {
		std::cerr << "Linking failed" << std::endl;
		return 1;
	}

	// Clean up the intermediate (object or bitcode) file.
	std::remove(intermediate.c_str());

	// Clear the OSC 9;4 progress indicator before either handing the
	// terminal to the child process (test/run mode) or printing the
	// success line, so the host terminal stops showing "in progress".
	progress.stop();

	if (testMode || runFlag) {
		std::string runCmd = "./" + outputName;
		int exitCode = system(runCmd.c_str());

		// Clean up executable after running
		std::remove(outputName.c_str());

// Extract actual exit code (system() returns encoded status)
#ifdef _WIN32
		return exitCode;
#else
		return WEXITSTATUS(exitCode);
#endif
	}

	std::cout << "Compilation successful: " << outputName << std::endl;
	return 0;
}

static std::vector<std::string> collectJamFiles(const std::string &dir) {
	std::vector<std::string> files;
	std::error_code ec;
	for (auto it = std::filesystem::recursive_directory_iterator(dir, ec);
	     it != std::filesystem::recursive_directory_iterator();
	     it.increment(ec)) {
		if (ec) break;
		const auto &entry = *it;
		if (entry.is_regular_file() && entry.path().extension() == ".jam") {
			files.push_back(entry.path().string());
		}
	}
	std::sort(files.begin(), files.end());
	return files;
}

static void printHelp(const char *prog) {
	std::cout
	    << "Usage: " << prog
	    << " [OPTIONS] <file|directory>\n"
	       "       "
	    << prog
	    << " run [LINKER-FLAGS] <file>\n"
	       "       "
	    << prog
	    << " test [<file|directory>]\n"
	       "\n"
	       "Subcommands:\n"
	       "  run             Compile, run, and clean up the executable. "
	       "Only linker\n"
	       "                  flags (-l<name>) may accompany it.\n"
	       "  test            Test mode: compile test functions and "
	       "run them\n"
	       "\n"
	       "Options:\n"
	       "  -C opt-level=N  Optimization level, Default is 0\n"
	       "                    0  no optimizations\n"
	       "                    1  basic optimizations\n"
	       "                    2  LLVM default (-O2)\n"
	       "                    3  aggressive (-O3)\n"
	       "                    s  optimize for size (-Os)\n"
	       "                    z  aggressively optimize for size "
	       "(-Oz)\n"
	       "  -C lto=MODE     Link-time optimization. Default is "
	       "`off`. Accepts:\n"
	       "                    off   regular object file (no LTO)\n"
	       "                    thin  ThinLTO bitcode — fast, "
	       "parallel link\n"
	       "                    fat   full LTO bitcode — slowest "
	       "link, most opt\n"
	       "  -C strip=MODE   Symbol / debug-info stripping. Default "
	       "is `none`.\n"
	       "                    none       keep all symbols & debug "
	       "info\n"
	       "                    debuginfo  strip DWARF / debug "
	       "sections only\n"
	       "                    symbols    strip debug + local "
	       "symbols\n"
	       "  --emit-ir       Print LLVM IR to stdout\n"
	       "  --target-info   Show host target info (arch, triple, "
	       "pointer size, ...)\n"
	       "  -o <name>       Output binary name (default: output)\n"
	       "  --std-path <dir>\n"
	       "                  Override the standard-library root used to\n"
	       "                  Takes precedence over the JAM_STD_PATH env var.\n"
	       "  -l<name>, --library <name>\n"
	       "                  Link against system library <name>\n"
	       "  -h, --help      Show this help and exit\n"
	       "\n"
	       "Examples:\n"
	       "  "
	    << prog
	    << " hello.jam                 # compile to ./output\n"
	       "  "
	    << prog
	    << " run hello.jam             # compile and run\n"
	       "  "
	    << prog
	    << " run -lncurses tetris.jam  # compile, link with ncurses, run\n"
	       "  "
	    << prog
	    << " test                      # run tests in cwd (recursive)\n"
	       "  "
	    << prog
	    << " test tests/unit           # run tests under tests/unit\n"
	       "  "
	    << prog << " test tests/unit/foo.jam   # run tests in a single file\n";
}

static bool fileHasTests(const std::string &path) {
	// Cheap substring scan — avoids lexing files with no `tfn` declarations.
	std::ifstream f(path);
	if (!f.is_open()) return false;
	std::stringstream ss;
	ss << f.rdbuf();
	const std::string src = ss.str();
	// Match `tfn` at start of a line or after whitespace, followed by
	// space/ident char.
	size_t pos = 0;
	while ((pos = src.find("tfn", pos)) != std::string::npos) {
		bool startOk = (pos == 0) ||
		               std::isspace(static_cast<unsigned char>(src[pos - 1]));
		bool endOk = (pos + 3 < src.size()) &&
		             (std::isspace(static_cast<unsigned char>(src[pos + 3])) ||
		              src[pos + 3] == '(');
		if (startOk && endOk) return true;
		pos += 3;
	}
	return false;
}

int main(int argc, char *argv[]) {
	// Parse command line arguments
	bool runFlag = false;
	bool showTarget = false;
	bool emitIR = false;
	bool testMode = false;
	JamOptLevel optLevel = JAM_OPT_NONE;
	JamLTO lto = JAM_LTO_OFF;
	JamStrip strip = JAM_STRIP_NONE;
	std::string filename;
	std::string outputName = "output";
	std::string stdPathOverride;
	std::vector<std::string> linkLibs;

	if (argc < 2) {
		printHelp(argv[0]);
		return 1;
	}

	// Parse subcommand + flags. `run` and `test` are subcommands; everything
	// else is either a flag or the filename. When `run` is in effect, only
	// linker flags (`-l<name>`, `-l <name>`, `--library <name>`) are
	// permitted alongside it.
	for (int i = 1; i < argc; i++) {
		std::string arg = argv[i];
		if (arg == "run") {
			runFlag = true;
			continue;
		}
		if (arg == "test") {
			testMode = true;
			continue;
		}
		// Linker flags — accepted in every mode.
		if ((arg == "-l" || arg == "--library") && i + 1 < argc) {
			linkLibs.push_back(argv[++i]);
			continue;
		}
		if (arg.length() > 2 && arg.substr(0, 2) == "-l") {
			linkLibs.push_back(arg.substr(2));
			continue;
		}
		// Friendly transition error: `--release` and `--release-small` were
		// the old optimization flags. Direct users to the new -C syntax
		// instead of letting the arg flow through as a phantom filename.
		if (arg == "--release") {
			std::cerr << "Error: `--release` was removed; use `-C "
			             "opt-level=3` instead"
			          << std::endl;
			return 1;
		}
		if (arg == "--release-small") {
			std::cerr << "Error: `--release-small` was removed; use `-C "
			             "opt-level=z` instead"
			          << std::endl;
			return 1;
		}
		// `-C key=value` codegen options, modeled after `rustc -C`. Accepts
		// both `-C key=value` (space) and `-Ckey=value` (no space) like
		// rustc. Default is debug (~30× faster compile).
		//
		// Supported keys:
		//   opt-level={0,1,2,3,s,z} — LLVM IR + codegen optimization level.
		//                              `s` = -Os, `z` = -Oz.
		{
			std::string codegenArg;
			if (arg == "-C" && i + 1 < argc) {
				codegenArg = argv[++i];
			} else if (arg.length() > 2 && arg.substr(0, 2) == "-C") {
				codegenArg = arg.substr(2);
			}
			if (!codegenArg.empty()) {
				auto eq = codegenArg.find('=');
				if (eq == std::string::npos) {
					std::cerr << "Error: -C expects key=value, got `"
					          << codegenArg << "`" << std::endl;
					return 1;
				}
				std::string key = codegenArg.substr(0, eq);
				std::string value = codegenArg.substr(eq + 1);
				if (key == "opt-level") {
					if (value == "0") optLevel = JAM_OPT_NONE;
					else if (value == "1") optLevel = JAM_OPT_LESS;
					else if (value == "2") optLevel = JAM_OPT_DEFAULT;
					else if (value == "3") optLevel = JAM_OPT_AGGRESSIVE;
					else if (value == "s") optLevel = JAM_OPT_SIZE;
					else if (value == "z") optLevel = JAM_OPT_SMALL;
					else {
						std::cerr << "Error: -C opt-level expects one of "
						             "0|1|2|3|s|z, got `"
						          << value << "`" << std::endl;
						return 1;
					}
				} else if (key == "lto") {
					if (value == "off" || value == "false" || value == "no") {
						lto = JAM_LTO_OFF;
					} else if (value == "thin") {
						lto = JAM_LTO_THIN;
					} else if (value == "fat" || value == "full" ||
					           value == "true" || value == "yes") {
						lto = JAM_LTO_FAT;
					} else {
						std::cerr << "Error: -C lto expects one of "
						             "off|thin|fat, got `"
						          << value << "`" << std::endl;
						return 1;
					}
				} else if (key == "strip") {
					if (value == "none" || value == "off" || value == "false" ||
					    value == "no") {
						strip = JAM_STRIP_NONE;
					} else if (value == "debuginfo") {
						strip = JAM_STRIP_DEBUGINFO;
					} else if (value == "symbols") {
						strip = JAM_STRIP_SYMBOLS;
					} else {
						std::cerr << "Error: -C strip expects one of "
						             "none|debuginfo|symbols, got `"
						          << value << "`" << std::endl;
						return 1;
					}
				} else {
					std::cerr << "Error: unknown -C key `" << key
					          << "` (supported: opt-level, lto, strip)"
					          << std::endl;
					return 1;
				}
				continue;
			}
		}
		// Inside `run`, anything else flag-shaped is an error.
		if (runFlag) {
			if (!arg.empty() && arg[0] == '-') {
				std::cerr << "Error: `run` only accepts linker flags "
				             "(-l<name>, -l <name>, --library <name>); got `"
				          << arg << "`" << std::endl;
				return 1;
			}
			if (!filename.empty()) {
				std::cerr << "Error: `run` accepts only one source file; got `"
				          << arg << "` after `" << filename << "`" << std::endl;
				return 1;
			}
			filename = arg;
			continue;
		}
		// Compile-only / test-mode flags.
		if (arg == "--help" || arg == "-h") {
			printHelp(argv[0]);
			return 0;
		}
		if (arg == "--target-info") {
			showTarget = true;
			continue;
		}
		if (arg == "--emit-ir") {
			emitIR = true;
			continue;
		}
		if (arg == "-o" && i + 1 < argc) {
			outputName = argv[++i];
			continue;
		}
		if (arg == "--std-path" && i + 1 < argc) {
			stdPathOverride = argv[++i];
			continue;
		}
		// Positional: source file or directory. Flags may follow it
		// (`./jam.out foo.jam --emit-ir` is valid); seeing a second
		// positional is an error.
		if (!filename.empty()) {
			std::cerr << "Error: unexpected extra argument `" << arg
			          << "` (already have `" << filename << "`)" << std::endl;
			return 1;
		}
		filename = arg;
	}

	// `jam test` with no path means "run every test under cwd".
	if (testMode && filename.empty()) { filename = "."; }

	if (filename.empty()) {
		std::cerr << "Error: No input file specified. Run `" << argv[0]
		          << " --help` for usage." << std::endl;
		return 1;
	}

	if (!stdPathOverride.empty()) { setStdPathOverride(stdPathOverride); }
	jam::Target target = jam::Target::getHostTarget();
	jam::CAbi cabi(target);

	if (showTarget) {
		std::cout << "Target Information:" << std::endl;
		std::cout << "  Name: " << target.getName() << std::endl;
		std::cout << "  Triple: " << target.toLLVMTriple() << std::endl;
		std::cout << "  Pointer size: " << target.getPointerSize() << " bytes"
		          << std::endl;
		std::cout << "  Libc: " << target.getLibCName() << std::endl;
		std::cout << "  Requires PIC: " << (target.requiresPIC() ? "yes" : "no")
		          << std::endl;
		std::cout << "  Requires PIE: " << (target.requiresPIE() ? "yes" : "no")
		          << std::endl;
		std::cout << "  Uses C ABI: " << (target.usesCabi() ? "yes" : "no")
		          << std::endl;
		std::cout << std::endl;
	}

	// Initialize LLVM using wrapper (once, even across multiple files)
	JamLLVMInitializeNativeTarget();
	JamLLVMInitializeNativeAsmPrinter();
	JamLLVMInitializeNativeAsmParser();

	// Directory input: only meaningful with `test`. Recursively discover .jam
	// files, run each one's tests, aggregate results.
	if (std::filesystem::is_directory(filename)) {
		if (!testMode) {
			std::cerr
			    << "Error: directory input is only supported with `test` (got '"
			    << filename << "')" << std::endl;
			return 1;
		}

		std::vector<std::string> files = collectJamFiles(filename);
		if (files.empty()) {
			std::cout << "No .jam files found under " << filename << std::endl;
			return 0;
		}

		int passed = 0;
		int failed = 0;
		int skipped = 0;
		for (const auto &f : files) {
			if (!fileHasTests(f)) {
				skipped++;
				continue;
			}
			std::cout << std::endl << "@" << f << std::endl;
			std::filesystem::path p(f);
			std::string perFileOutput = "jam_test_" + p.stem().string();
			int rc = compileAndRun(f, perFileOutput, runFlag, emitIR, testMode,
			                       optLevel, lto, strip, linkLibs);
			if (rc != 0) failed++;
			else passed++;
		}

		std::cout << std::endl;
		std::cout << "Summary: " << passed << " file(s) passed, " << failed
		          << " file(s) failed, " << skipped
		          << " file(s) without tests, " << files.size()
		          << " file(s) scanned" << std::endl;
		return failed == 0 ? 0 : 1;
	}

	// Catch compile-time exceptions cleanly so the user sees a single-line
	// error instead of a stack trace + abort. Exceptions reach here from
	// codegen paths that detect impossible inputs (e.g. a generic
	// instantiation referencing a method the concrete type doesn't have).
	try {
		return compileAndRun(filename, outputName, runFlag, emitIR, testMode,
		                     optLevel, lto, strip, linkLibs);
	} catch (const std::exception &e) {
		std::cerr << filename << ": error: " << e.what() << std::endl;
		return 1;
	}
}
