/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <memory>
#include <map>

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"

#include "lexer.h"
#include "parser.h"
#include "ast.h"
#include "target.h"
#include "cabi.h"

int main(int argc, char* argv[]) {
    // Parse command line arguments
    bool runFlag = false;
    bool showTarget = false;
    std::string filename;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " [--run] [--target-info] <filename>" << std::endl;
        return 1;
    }
    
    // Parse flags
    int fileArgIndex = 1;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--run") {
            runFlag = true;
        } else if (arg == "--target-info") {
            showTarget = true;
        } else {
            filename = arg;
            break;
        }
    }
    
    if (filename.empty()) {
        std::cerr << "Error: No input file specified" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [--run] [--target-info] <filename>" << std::endl;
        return 1;
    }
    
    // Get target information
    jam::Target target = jam::Target::getHostTarget();
    jam::CAbi cabi(target);
    
    if (showTarget) {
        std::cout << "Target Information:" << std::endl;
        std::cout << "  Name: " << target.getName() << std::endl;
        std::cout << "  Triple: " << target.toLLVMTriple() << std::endl;
        std::cout << "  Pointer size: " << target.getPointerSize() << " bytes" << std::endl;
        std::cout << "  Libc: " << target.getLibCName() << std::endl;
        std::cout << "  Requires PIC: " << (target.requiresPIC() ? "yes" : "no") << std::endl;
        std::cout << "  Requires PIE: " << (target.requiresPIE() ? "yes" : "no") << std::endl;
        std::cout << "  Uses C ABI: " << (target.usesCabi() ? "yes" : "no") << std::endl;
        std::cout << std::endl;
    }
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();

    // Initialize LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Create a LLVM context and module
    llvm::LLVMContext Context;
    std::unique_ptr<llvm::Module> TheModule = std::make_unique<llvm::Module>("my cool compiler", Context);

    // Create a builder for IR generation
    llvm::IRBuilder<> Builder(Context);

    // Tokenize the source code
    Lexer lexer(source);
    std::vector<Token> tokens = lexer.scanTokens();

    // Parse the tokens into an AST
    Parser parser(tokens);
    std::vector<std::unique_ptr<FunctionAST>> functions = parser.parse();

    // Create a map to store variable values
    std::map<std::string, llvm::Value*> NamedValues;

    // Generate code from the AST
    for (auto& function : functions) {
        function->codegen(Builder, TheModule.get(), NamedValues);
    }

    if (runFlag) {
        // Execute the code directly using LLVM JIT
        std::cout << "Running Jam program..." << std::endl;
        
        // Create execution engine
        std::string ErrStr;
        llvm::ExecutionEngine* EE = llvm::EngineBuilder(std::move(TheModule))
            .setErrorStr(&ErrStr)
            .setEngineKind(llvm::EngineKind::JIT)
            .create();
        
        if (!EE) {
            std::cerr << "Failed to create execution engine: " << ErrStr << std::endl;
            return 1;
        }
        
        // Find the main function
        llvm::Function* MainFn = EE->FindFunctionNamed("main");
        if (!MainFn) {
            std::cerr << "Error: No main function found" << std::endl;
            delete EE;
            return 1;
        }
        
        // Execute the main function
        std::vector<llvm::GenericValue> Args;
        llvm::GenericValue Result = EE->runFunction(MainFn, Args);
        
        // Print the result if main returns a value
        if (!MainFn->getReturnType()->isVoidTy()) {
            std::cout << std::endl << "Program exited with code: " << Result.IntVal.getZExtValue() << std::endl;
        } else {
            std::cout << std::endl << "Program completed successfully." << std::endl;
        }
        
        delete EE;
        return 0;
    } else {
        // Print out the generated LLVM IR
        std::string output;
        llvm::raw_string_ostream out(output);
        TheModule->print(out, nullptr);
        std::cout << output;

        // Now create the output binary
        std::string TargetTriple = llvm::sys::getDefaultTargetTriple();
        TheModule->setTargetTriple(TargetTriple);

        std::string Error;
        const llvm::Target* Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);

        if (!Target) {
            std::cerr << "Failed to get target: " << Error << std::endl;
            return 1;
        }

        llvm::TargetOptions opt;
        auto RM = std::optional<llvm::Reloc::Model>();
        auto TargetMachine = Target->createTargetMachine(TargetTriple, "generic", "", opt, RM);

        TheModule->setDataLayout(TargetMachine->createDataLayout());

        std::string ObjectFilename = "output.o";
        std::error_code EC;
        llvm::raw_fd_ostream dest(ObjectFilename, EC, llvm::sys::fs::OF_None);

        if (EC) {
            std::cerr << "Could not open file: " << EC.message() << std::endl;
            return 1;
        }

        llvm::legacy::PassManager pass;
        if (TargetMachine->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
            std::cerr << "TargetMachine can't emit a file of this type" << std::endl;
            return 1;
        }

        pass.run(*TheModule);
        dest.close();

        // Finish up by creating an executable using system compiler
        std::string cmd = "clang " + ObjectFilename + " -o output";
        system(cmd.c_str());

        std::cout << "Compilation completed successfully." << std::endl;
        return 0;
    }
}
