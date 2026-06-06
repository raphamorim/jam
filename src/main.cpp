/*
 * Copyright (c) 2026-present Raphael Amorim
 *
 * This file is part of jam.
 * Licensed under the Apache License, Version 2.0 with LLVM Exceptions.
 */

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#endif

#include "analyzer.h"
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

	Parser parser(tokens, lexer.sourceBuffer(), codegenCtx.getTypePool(),
	              codegenCtx.getStringPool(), codegenCtx.getNodeStore(),
	              &codegenCtx.diagnostics(), filename);
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

	// Walk a `import("base").seg.seg` chain through pub-import re-
	// exports to the final target module. Each segment names a
	// `pub const X = import(...)` alias in the preceding module; we
	// recurse so a re-export that itself walks a chain composes.
	auto resolveImportChain =
	    [&](const std::string &basePath, const std::vector<std::string> &chain,
	        auto &self) -> std::pair<std::string, ModuleAST *> {
		std::string curPath = basePath;
		ModuleAST *curMod = resolver.getOrLoadModule(curPath);
		if (!curMod) return {curPath, nullptr};
		for (const auto &seg : chain) {
			const ImportDeclAST *re = nullptr;
			for (auto &imp : curMod->Imports) {
				if (imp->isPub && imp->Name == seg) {
					re = imp.get();
					break;
				}
			}
			if (!re) return {curPath + "." + seg, nullptr};
			auto sub = self(re->Path, re->chain, self);
			if (!sub.second) return sub;
			curPath = sub.first;
			curMod = sub.second;
		}
		return {curPath, curMod};
	};

	symbolTable.registerBuiltinSymbol("test", "assert");
	for (auto &import : module->Imports) {
		if (import->Path == "test") { continue; }

		auto resolved =
		    resolveImportChain(import->Path, import->chain, resolveImportChain);
		ModuleAST *importedModule = resolved.second;
		if (!importedModule) {
			std::cerr << "Error: Failed to load module: " << resolved.first
			          << std::endl;
			return 1;
		}
		// Rewrite Path to the chain-resolved canonical so every later
		// pass (handle registration, namespace build) sees one path.
		import->Path = resolved.first;
		import->chain.clear();

		symbolTable.registerModule(import->Path, importedModule);
	}

	for (auto &destImport : module->DestructuringImports) {
		auto resolved = resolveImportChain(destImport->Path, destImport->chain,
		                                   resolveImportChain);
		ModuleAST *importedModule = resolved.second;
		if (!importedModule) {
			std::cerr << "Error: Failed to load module: " << resolved.first
			          << std::endl;
			return 1;
		}
		destImport->Path = resolved.first;
		destImport->chain.clear();

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

	// Populate the demand-driven DeclTable: one Decl per top-level
	// binding across the main module + every imported module. The
	// analyzer consults this table for cycle detection when codegen
	// resolves a generic call, a Named type, or any other cross-decl
	// reference. Step 4 will make this the *only* source of truth and
	// delete the parallel functionAsts / struct registry tables; for
	// now we populate it in parallel so the existing eager pipeline
	// stays unchanged.
	auto registerTopLevelDecls = [&](ModuleAST *m, bool publicOnly) {
		auto setSrc = [&](jam::Decl &d, const std::string &name) {
			(void)name;
			d.file = codegenCtx.currentFile();
			// Line populated when AST nodes carry per-node line; for
			// the top-level decl we use 0 (no specific line) for
			// now.
		};
		for (auto &fn : m->Functions) {
			if (publicOnly && !fn->isPub) continue;
			jam::DeclIndex idx = codegenCtx.declTable().create(
			    jam::DeclKind::Function, fn->Name);
			auto &d = codegenCtx.declTable().get(idx);
			d.fnAst = fn.get();
			setSrc(d, fn->Name);
		}
		for (auto &s : m->Structs) {
			if (publicOnly && !s->isPub) continue;
			jam::DeclIndex idx =
			    codegenCtx.declTable().create(jam::DeclKind::Struct, s->Name);
			auto &d = codegenCtx.declTable().get(idx);
			d.structAst = s.get();
			setSrc(d, s->Name);
		}
		for (auto &e : m->Enums) {
			if (publicOnly && !e->isPub) continue;
			jam::DeclIndex idx =
			    codegenCtx.declTable().create(jam::DeclKind::Enum, e->Name);
			auto &d = codegenCtx.declTable().get(idx);
			d.enumAst = e.get();
			setSrc(d, e->Name);
		}
		for (auto &u : m->Unions) {
			if (publicOnly && !u->isPub) continue;
			jam::DeclIndex idx =
			    codegenCtx.declTable().create(jam::DeclKind::Union, u->Name);
			auto &d = codegenCtx.declTable().get(idx);
			d.unionAst = u.get();
			setSrc(d, u->Name);
		}
		for (auto &c : m->Consts) {
			if (publicOnly && !c->isPub) continue;
			jam::DeclIndex idx =
			    codegenCtx.declTable().create(jam::DeclKind::Const, c->Name);
			auto &d = codegenCtx.declTable().get(idx);
			d.constAst = c.get();
			setSrc(d, c->Name);
		}
	};
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		registerTopLevelDecls(importedModule.get(), /*publicOnly=*/true);
	}
	registerTopLevelDecls(module.get(), /*publicOnly=*/false);

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
		declareStructs(importedModule.get(), /*publicOnly=*/true);
		declareUnions(importedModule.get(), /*publicOnly=*/true);
		declareEnums(importedModule.get(), /*publicOnly=*/true);
		declareEnumLLVMTypes(importedModule.get());
	}
	declareStructs(module.get(), /*publicOnly=*/false);
	declareUnions(module.get(), /*publicOnly=*/false);
	declareEnums(module.get(), /*publicOnly=*/false);
	declareEnumLLVMTypes(module.get());

	// Register imported pub functions + handle-import metadata + emit
	// their LLVM prototypes BEFORE filling struct/union/enum bodies.
	// A struct field typed as a generic instantiation
	// (`inner: Vec(i32)`) drives `getLLVMType` to resolve the
	// GenericCall, which instantiates Vec for i32 and codegens its
	// methods. Those methods call externs from the same imported
	// module (e.g. `malloc`), so the externs need prototypes
	// declared by the time the instantiation runs.
	//
	// Split into two passes so iteration order over loaded modules
	// doesn't matter: Pass A registers every module's pub fns +
	// ModuleNamespace into the global registry; Pass B emits the LLVM
	// prototypes. Without the split, if `bus.jam` is iterated before
	// `std/collections` (unordered_map order), bus's `dma: Vec(u32)`
	// field triggers Vec instantiation whose body calls `malloc` —
	// but std/collections's `malloc` hasn't been globally registered
	// yet, so the body fails to compile.
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		JamCodegenContext::ModuleNamespace ns;
		ns.path = path;
		for (auto &func : importedModule->Functions) {
			// All functions (pub + private) go into the per-module ns
			// so a body in this module can resolve a sibling helper
			// regardless of pub status. Only pub ones leak into the
			// global flat map.
			ns.functions[func->Name] = func.get();
			if (func->isPub) {
				codegenCtx.registerFunctionAST(func->Name, func.get());
				// Eagerly declare LLVM prototypes for pub-extern fns
				// (libc allocator, write, putchar, …). A generic
				// instantiation in Pass B may emit a Call to one of
				// these BEFORE the iteration reaches its defining
				// module — without an eager declaration, the call
				// lowers to "unknown callee `malloc`". Non-extern
				// pub fns still need their full signature to be
				// resolved against module-level types, so we defer
				// those to Pass B.
				if (func->isExtern && !func->isGeneric()) {
					JirFunction jfn = astgenMetadata(*func, codegenCtx);
					jfn.name =
					    mangledFunctionName(*func, codegenCtx.getTypePool(),
					                        codegenCtx.getStringPool());
					jirDeclarePrototype(jfn, codegenCtx);
				}
			}
		}
		for (auto &s : importedModule->Structs) {
			if (s->isPub) {
				ns.types[s->Name] = codegenCtx.getTypePool().internNamed(
				    codegenCtx.getStringPool().intern(s->Name));
			}
		}
		for (auto &e : importedModule->Enums) {
			if (e->isPub) {
				ns.types[e->Name] = codegenCtx.getTypePool().internNamed(
				    codegenCtx.getStringPool().intern(e->Name));
			}
		}
		for (auto &u : importedModule->Unions) {
			if (u->isPub) {
				ns.types[u->Name] = codegenCtx.getTypePool().internNamed(
				    codegenCtx.getStringPool().intern(u->Name));
			}
		}
		for (auto &reexport : importedModule->Imports) {
			if (!reexport->isPub) continue;
			if (reexport->Path == "test") continue;
			TypeIdx modTy = codegenCtx.getTypePool().internModule(
			    codegenCtx.getStringPool().intern(reexport->Path));
			ns.moduleAliases[reexport->Name] = modTy;
		}
		codegenCtx.registerModuleNamespace(std::move(ns));
	}
	// Pass B: now that every module's pub fns are visible, emit LLVM
	// prototypes. Any generic instantiation triggered by a parameter
	// or return type resolves its body's externs against the
	// already-populated global registry — or, via getFunctionAST's
	// fallback, against the generic's defining-module namespace.
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		for (auto &func : importedModule->Functions) {
			if (func->isGeneric()) continue;
			// Pub fns need prototypes so callers can call them. Private
			// helpers also need them so the pub bodies above that
			// reference them (via defining-module-scope lookup) resolve
			// at LLVM codegen time.
			JirFunction jfn = astgenMetadata(*func, codegenCtx);
			jfn.name = mangledFunctionName(*func, codegenCtx.getTypePool(),
			                               codegenCtx.getStringPool());
			jirDeclarePrototype(jfn, codegenCtx);
		}
	}
	// Register every flat `handle.X` mapping for a given (handle name,
	// resolved module). Shared by direct-import bindings and module-
	// valued destructuring bindings (`const {fmt} = import("std");`).
	// `owner` is the module whose imports these are -- and the scope key its
	// bodies resolve against. "" is the entry module: its bodies compile with
	// an empty bodyModuleStack_ (which maps to the "" key), and entry-defined
	// generics carry an empty modulePath (same key). EVERY module -- entry
	// included -- gets its OWN scope in moduleImports_, so a body resolves its
	// qualified imports against its own module and never inherits another's.
	// There is deliberately NO cross-module fallback (see getImportHandle): a
	// qualified name absent from the declaring module's imports is an error,
	// not a lookup in the entry/root module.
	auto registerHandleFlats = [&](const std::string &handle,
	                               const std::string &modulePath,
	                               ModuleAST *importedModule,
	                               const std::string &owner) {
		codegenCtx.registerScopedHandle(owner, handle, modulePath);
		auto regFn = [&](const std::string &key, const FunctionAST *fn) {
			codegenCtx.registerScopedHandleFn(owner, key, fn);
		};
		auto regPriv = [&](const std::string &name) {
			codegenCtx.registerScopedPrivateName(owner, handle, name);
		};
		auto aliasNamed = [&](const std::string &bare) {
			TypeIdx target = codegenCtx.getTypePool().internNamed(
			    codegenCtx.getStringPool().intern(bare));
			codegenCtx.registerScopedTypeAlias(owner, handle + "." + bare,
			                                   target);
		};
		for (auto &func : importedModule->Functions) {
			if (func->isPub) regFn(handle + "." + func->Name, func.get());
			else regPriv(func->Name);
		}
		for (auto &s : importedModule->Structs) {
			if (s->isPub) {
				aliasNamed(s->Name);
				for (auto &m : s->Methods) {
					if (m->isPub)
						regFn(handle + "." + s->Name + "." + m->Name, m.get());
				}
			} else {
				regPriv(s->Name);
			}
		}
		for (auto &e : importedModule->Enums) {
			if (e->isPub) aliasNamed(e->Name);
			else regPriv(e->Name);
		}
		for (auto &u : importedModule->Unions) {
			if (u->isPub) aliasNamed(u->Name);
			else regPriv(u->Name);
		}
	};

	for (auto &import : module->Imports) {
		if (import->Path == "test") continue;
		ModuleAST *importedModule = resolver.getOrLoadModule(import->Path);
		if (!importedModule) continue;
		registerHandleFlats(import->Name, import->Path, importedModule, "");
	}

	// Destructured names that bind a re-exported module value — treat
	// them as if the user had written `const X = import("...").X;` so
	// `X.member` resolves through the standard handle-flat tables.
	for (auto &destImport : module->DestructuringImports) {
		if (destImport->Path == "test") continue;
		ModuleAST *src = resolver.getOrLoadModule(destImport->Path);
		if (!src) continue;
		for (const auto &name : destImport->Names) {
			const ImportDeclAST *re = nullptr;
			for (auto &imp : src->Imports) {
				if (imp->isPub && imp->Name == name) {
					re = imp.get();
					break;
				}
			}
			if (!re) continue;
			auto resolved =
			    resolveImportChain(re->Path, re->chain, resolveImportChain);
			if (!resolved.second) continue;
			registerHandleFlats(name, resolved.first, resolved.second, "");
		}
	}

	// Per-module import bindings for every OTHER loaded module, scoped to that
	// module's path. So when an imported pub/cfn body is compiled later (under
	// pushBodyModule(modPath) in the imported-body pass), its qualified imports
	// (`std.fmt.print`) resolve against ITS module's imports, not the entry
	// module's -- a body resolves names against its own declaring module.
	// Mirrors the two entry-module passes above, but walks each loaded module's
	// own Imports/DestructuringImports.
	for (const auto &[modPath, loadedModule] : resolver.getLoadedModules()) {
		for (auto &imp : loadedModule->Imports) {
			if (imp->Path == "test") continue;
			ModuleAST *target = resolver.getOrLoadModule(imp->Path);
			if (!target) continue;
			registerHandleFlats(imp->Name, imp->Path, target, modPath);
		}
		for (auto &destImp : loadedModule->DestructuringImports) {
			if (destImp->Path == "test") continue;
			ModuleAST *src = resolver.getOrLoadModule(destImp->Path);
			if (!src) continue;
			for (const auto &name : destImp->Names) {
				const ImportDeclAST *re = nullptr;
				for (auto &imp : src->Imports) {
					if (imp->isPub && imp->Name == name) {
						re = imp.get();
						break;
					}
				}
				if (!re) continue;
				auto resolved =
				    resolveImportChain(re->Path, re->chain, resolveImportChain);
				if (!resolved.second) continue;
				registerHandleFlats(name, resolved.first, resolved.second,
				                    modPath);
			}
		}
	}

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
		registerConsts(importedModule.get());
	}
	registerConsts(module.get());

	// Single demand-driven body-fill pass: walk every Struct/Enum/
	// Union decl in the DeclTable and ask the analyzer to materialise
	// it. The per-kind `resolveTypeFields*` functions are re-entrant
	// — when a struct's field references another struct/enum/union,
	// the field walk's ensure*Body call fills that dependency
	// transitively. The publicOnly filtering that the per-module
	// fill*Bodies lambdas used to do isn't needed any more because
	// `registerTopLevelDecls` already applied that filter when
	// populating the DeclTable.
	//
	// Runs AFTER module-const registration: a field typed `[SIZE]u8`
	// comptime-folds SIZE against the const scope when getLLVMType
	// resolves the deferred array length.
	auto &dt = codegenCtx.declTable();
	for (std::size_t i = 1; i < dt.all().size(); ++i) {
		jam::DeclIndex idx = static_cast<jam::DeclIndex>(i);
		const jam::Decl &dr = dt.get(idx);
		if (dr.kind == jam::DeclKind::Struct ||
		    dr.kind == jam::DeclKind::Enum || dr.kind == jam::DeclKind::Union) {
			codegenCtx.analyzer().ensureDeclAnalyzed(idx);
		}
	}

	// Two-pass codegen: declare every function's prototype first, then
	// emit bodies. Without this, calling a function defined later in the
	// file (or another module) would fail with "Unknown function". The
	// two-pass shape lets source read naturally top-down (main on top,
	// helpers below) without a manual "forward declarations" section.
	//
	// Pass 1a (imported pub fn prototypes) and the handle-import
	// metadata loop ran earlier — see above declareStructs — because
	// struct-field generic instantiation needs them to be in place.

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
	// Validate + register every top-level struct's methods. Imported
	// modules' structs need the same treatment because struct methods
	// aren't otherwise re-discovered when only their decl is aliased
	// across module boundaries. Without this, calls like
	// `someImportedStruct.method(...)` would fail with "unknown method".
	// For imported modules we honor pub: a non-pub method on a struct
	// in module B is invisible to module A's call sites.
	auto registerStructMethods = [&](ModuleAST *m, bool publicOnly) -> int {
		for (auto &s : m->Structs) {
			for (auto &meth : s->Methods) {
				// `cfn` methods (drop / default / at / setAt / len) are
				// compiler-synthesized hooks that may be invoked from OTHER
				// modules (e.g. `drop` fires at a scope exit in the importing
				// module), so they must be declared + codegen'd even when not
				// `pub`. Plain non-pub methods stay module-private.
				if (publicOnly && !meth->isPub && !meth->isCfn) continue;
				// `cfn`-marked methods (drop / default / at / …) opt
				// in to compiler-synthesized calls and must match the
				// expected signature for their name. Plain `fn`
				// methods are ordinary instance methods — no
				// signature constraints, no rejection by name.
				if (meth->isCfn && meth->Name == "default") {
					if (!meth->Args.empty()) {
						std::cerr << filename
						          << ": error: cfn `default` on struct `"
						          << s->Name << "` must take no parameters\n";
						return 1;
					}
					std::string retStruct = resolveStructName(meth->ReturnType);
					if (retStruct != s->Name) {
						std::cerr << filename
						          << ": error: cfn `default` on struct `"
						          << s->Name << "` must return `Self` (got `"
						          << retStruct << "`)\n";
						return 1;
					}
				} else if (meth->isCfn && meth->Name == "drop") {
					if (meth->Args.empty() || meth->Args[0].Name != "self") {
						std::cerr
						    << filename << ": error: cfn `" << meth->Name
						    << "` on struct `" << s->Name
						    << "` must take `self` as its first parameter\n";
						return 1;
					}
					std::string selfStruct =
					    resolveStructName(meth->Args[0].Type);
					if (selfStruct != s->Name) {
						std::cerr << filename << ": error: cfn `" << meth->Name
						          << "` on struct `" << s->Name
						          << "` has self type `" << selfStruct
						          << "`; expected `" << s->Name << "`\n";
						return 1;
					}
				}
				{
					JirFunction jfn = astgenMetadata(*meth, codegenCtx);
					jfn.name =
					    mangledFunctionName(*meth, codegenCtx.getTypePool(),
					                        codegenCtx.getStringPool());
					jirDeclarePrototype(jfn, codegenCtx);
				}
				codegenCtx.registerFunctionAST(s->Name + "." + meth->Name,
				                               meth.get());
			}
		}
		return 0;
	};
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		int rc = registerStructMethods(importedModule.get(),
		                               /*publicOnly=*/true);
		if (rc != 0) return rc;
	}
	int rc = registerStructMethods(module.get(), /*publicOnly=*/false);
	if (rc != 0) return rc;

	// Build the drop registry up front so AstGen can read it via
	// `JamCodegenContext::getDropRegistry()` to track which `var`
	// bindings need their drop fn called at scope exit.
	jam::drops::DropRegistry dropRegistry = jam::drops::buildDropRegistry(
	    *module, codegenCtx.getTypePool(), codegenCtx.getStringPool());
	// Fold imported modules' drops into the registry: a drop site in the main
	// module must fire the destructor of a type defined and imported from
	// another module (e.g. `Bus.drop` in bus.jam, dropped in main.jam).
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		jam::drops::addDropCandidates(dropRegistry, *importedModule,
		                              codegenCtx.getTypePool(),
		                              codegenCtx.getStringPool());
	}
	codegenCtx.setDropRegistry(&dropRegistry);

	// Clone counterpart: `cfn clone(self: T) T` per type, top-level and
	// in-struct forms (CLONE_PLAN.md).
	jam::drops::CloneRegistry cloneRegistry;
	jam::drops::addCloneCandidates(cloneRegistry, *module,
	                               codegenCtx.getTypePool(),
	                               codegenCtx.getStringPool());
	for (const auto &[path, importedModule] : resolver.getLoadedModules()) {
		jam::drops::addCloneCandidates(cloneRegistry, *importedModule,
		                               codegenCtx.getTypePool(),
		                               codegenCtx.getStringPool());
	}
	codegenCtx.setCloneRegistry(&cloneRegistry);

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
		// Push the defining module so the bodies of `pub` functions
		// imported from it can resolve bare-name calls (e.g. a private
		// helper called from a pub fn) against that module's namespace,
		// not the caller's. Same mechanism the generic-instantiation
		// path uses — see JamCodegenContext::pushBodyModule.
		codegenCtx.pushBodyModule(path);
		// Attribute diagnostics from this module's bodies to ITS file, not the
		// entry file. The global NodeStore already carries the correct line;
		// only currentFile() (used by locOf) is entry-global, so an error in an
		// imported body must restore the right filename here.
		std::string prevFile = codegenCtx.currentFile();
		codegenCtx.setCurrentFile(path + ".jam");
		for (auto &func : importedModule->Functions) {
			if (func->isGeneric()) continue;
			if (func->isExtern) continue;
			// Emit bodies for ALL non-extern fns in the imported
			// module (pub + private). A pub fn that calls a private
			// helper needs the helper's body emitted in the same
			// compilation unit so LLVM has a definition to link.
			try {
				JirFunction jfn = astgenFunction(*func, codegenCtx);
				jfn.name = mangledFunctionName(*func, codegenCtx.getTypePool(),
				                               codegenCtx.getStringPool());
				jirFunctions.push_back(std::move(jfn));
			} catch (const AstGenAnalysisFail &) {
				// diagnostic already pushed
			}
		}
		for (auto &s : importedModule->Structs) {
			for (auto &m : s->Methods) {
				if (!m->isPub && !m->isCfn)
					continue;  // cfn hooks (drop/...) are invocable
					           // cross-module
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
		codegenCtx.setCurrentFile(prevFile);
		codegenCtx.popBodyModule();
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
			    if (k.kind == TypeKind::ArrayExpr) {
				    return cc->resolveArrayExpr(t);
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
			for (auto &fn : kv.second->Functions) {
				// Non-pub fns register too: the analysis sweep now
				// covers imported BODIES, and a pub fn's body calls its
				// module's private helpers — their modes must resolve
				// or moves through them go unseen. Flat-by-name, so a
				// private-helper name collision across modules resolves
				// last-wins (acceptable: modes rarely differ for
				// same-named helpers; a per-module registry is the
				// precise fix if it ever bites).
				fnRegistry[fn->Name] = fn.get();
			}
		}
		// Methods of generic struct-returning functions (`pub fn Vec(T:
		// type) type { return struct { fn push(value: move T) ... }; }`)
		// register under "GenericName.method". Parameter MODES don't
		// depend on T, so the un-instantiated FunctionAST is enough for
		// the mode-aware callsite analysis to see `v.push(c)` as a move.
		auto registerAnonMethods = [&](const ModuleAST *m) {
			const NodeStore &nsr = codegenCtx.getNodeStore();
			for (const auto &fn : m->Functions) {
				if (!fn->isGeneric()) continue;
				for (NodeIdx stmt : fn->Body) {
					const AstNode &rn = nsr.get(stmt);
					if (rn.tag != AstTag::Return) continue;
					if (rn.lhs == kNoNode) break;
					const AstNode &value =
					    nsr.get(static_cast<NodeIdx>(rn.lhs));
					if (value.tag != AstTag::StructExpr) break;
					uint32_t anonIdx = value.lhs;
					if (anonIdx >= sharedAnonStructs.size()) break;
					const StructDeclAST *anon =
					    sharedAnonStructs[anonIdx].get();
					for (const auto &mth : anon->Methods) {
						fnRegistry[fn->Name + "." + mth->Name] = mth.get();
					}
					break;
				}
			}
		};
		registerAnonMethods(module.get());
		for (const auto &kv : resolver.getLoadedModules()) {
			registerAnonMethods(kv.second.get());
		}

		// Enum-variant table: concrete enums by name, plus generic
		// enum factories (`pub fn Option(T: type) type { return enum
		// {...}; }`) under the factory fn's name. Lets the analyzer
		// treat `Maybe.Some(c)` / `Option(T).Some(c)` payload args as
		// the moves they are.
		jam::init_analysis::EnumVariantMap enumVariants;
		auto registerEnumVariants = [&](const ModuleAST *m) {
			for (const auto &e : m->Enums) {
				auto &set = enumVariants[e->Name];
				for (const auto &v : e->Variants) { set.insert(v.Name); }
			}
			const NodeStore &nsr = codegenCtx.getNodeStore();
			for (const auto &fn : m->Functions) {
				if (!fn->isGeneric()) continue;
				for (NodeIdx stmt : fn->Body) {
					const AstNode &rn = nsr.get(stmt);
					if (rn.tag != AstTag::Return) continue;
					if (rn.lhs == kNoNode) break;
					const AstNode &value =
					    nsr.get(static_cast<NodeIdx>(rn.lhs));
					if (value.tag != AstTag::EnumExpr) break;
					uint32_t anonIdx = value.lhs;
					if (anonIdx >= sharedAnonEnums.size()) break;
					auto &set = enumVariants[fn->Name];
					for (const auto &v : sharedAnonEnums[anonIdx]->Variants) {
						set.insert(v.Name);
					}
					break;
				}
			}
		};
		registerEnumVariants(module.get());
		for (const auto &kv : resolver.getLoadedModules()) {
			registerEnumVariants(kv.second.get());
		}

		auto runAnalysisIn = [&](FunctionAST *function,
		                         const std::string &file) {
			if (function->isExtern) return;
			auto diags = jam::init_analysis::analyze(
			    *function, codegenCtx.getNodeStore(),
			    codegenCtx.getStringPool(), tokens, &fnRegistry, &dropRegistry,
			    &codegenCtx.getTypePool(), &enumVariants);
			// Funnel each init-analysis diagnostic into the unified
			// `jam::Diagnostics` channel so they share the same
			// formatting / ordering as astgen errors.
			for (auto &d : diags) {
				jam::SrcLoc loc{file, d.line};
				codegenCtx.diagnostics().error(loc, d.message);
			}
		};
		auto runAnalysis = [&](FunctionAST *function) {
			runAnalysisIn(function, filename);
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
		// Imported modules get the same sweep, with diagnostics
		// attributed to the DEFINING file (NodeStore's per-node line
		// table is parse-time-stamped, so lines are already correct).
		// Generic factory bodies stay skipped here, same as the entry
		// module — their CLONES are the analyzable artifacts, and
		// analyzing clones is blocked on std APIs that duplicate
		// drop-bearing values by design (withCapacity's fill, get/at
		// copies); see TODO.md #12.
		for (const auto &kv : resolver.getLoadedModules()) {
			std::string importedFile = kv.first + ".jam";
			for (auto &fn : kv.second->Functions) {
				if (fn->isGeneric()) continue;
				runAnalysisIn(fn.get(), importedFile);
			}
			for (auto &s : kv.second->Structs) {
				for (auto &m : s->Methods) {
					runAnalysisIn(m.get(), importedFile);
				}
			}
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

	// `--emit-ir` is a "print IR and exit" mode — skipping the object
	// emit + link step matches clang's `-emit-llvm -S` / rustc's
	// `--emit=llvm-ir` / zig's `-femit-llvm-ir` behavior. Critically
	// it also dodges the default output name (`./output`) colliding
	// with the build tree's `output/` directory in this repo.
	if (emitIR) {
		char *irStr = JamLLVMPrintModuleToString(codegenCtx.getModule());
		std::cout << irStr;
		JamLLVMDisposeMessage(irStr);
		progress.stop();
		return 0;
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
		if (exitCode == -1) {
			std::cerr << "Failed to run " << outputName << std::endl;
			return 1;
		}
		// A signal-killed child (segfault, abort, …) has no exit status;
		// WEXITSTATUS on it reads garbage bits that decode to 0, which
		// would report a crashed test binary as "passed" — and since the
		// crash also loses the child's unflushed stdout, the failure
		// would be completely silent. Follow the shell convention:
		// 128 + signal number.
		if (WIFSIGNALED(exitCode)) {
			int sig = WTERMSIG(exitCode);
			std::cerr << outputName << " terminated by signal " << sig << " ("
			          << strsignal(sig) << ")" << std::endl;
			return 128 + sig;
		}
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

// JAM_VERSION_BASE pins the semver-ish prefix; the suffix is the git
// short SHA the binary was built from, baked in by the Makefile via
// `-D JAM_VERSION_SHA="..."`. Falls back to "unknown" when the build
// system doesn't supply one (out-of-tree, no .git, etc.).
#ifndef JAM_VERSION_BASE
#define JAM_VERSION_BASE "0.0.1"
#endif
#ifndef JAM_VERSION_SHA
#define JAM_VERSION_SHA "unknown"
#endif

static void printVersion() {
	std::cout << JAM_VERSION_BASE "-" JAM_VERSION_SHA "\n";
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
	       "  -V, --version   Print version and exit\n"
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
		if (arg == "--version" || arg == "-V") {
			printVersion();
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
