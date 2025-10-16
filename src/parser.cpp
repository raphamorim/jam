/*
 * Copyright (c) 2025 Raphael Amorim
 *
 * This file is part of jam, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#include "parser.h"
#include <stdexcept>

Parser::Parser(std::vector<Token> tokens) : tokens(std::move(tokens)) {}

Token Parser::peek() const {
    return tokens[current];
}

Token Parser::previous() const {
    return tokens[current - 1];
}

bool Parser::isAtEnd() const {
    return peek().type == TOK_EOF;
}

Token Parser::advance() {
    if (!isAtEnd()) current++;
    return previous();
}

bool Parser::check(TokenType type) const {
    if (isAtEnd()) return false;
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

void Parser::consume(TokenType type, const std::string& message) {
    if (check(type)) {
        advance();
        return;
    }

    throw std::runtime_error(message);
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
    if (match(TOK_NUMBER)) {
        return std::make_unique<NumberExprAST>(std::stoll(previous().lexeme));
    } else if (match(TOK_TRUE)) {
        return std::make_unique<BooleanExprAST>(true);
    } else if (match(TOK_FALSE)) {
        return std::make_unique<BooleanExprAST>(false);
    } else if (match(TOK_STRING_LITERAL)) {
        return std::make_unique<StringLiteralExprAST>(previous().lexeme);
    } else if (match(TOK_OPEN_PAREN)) {
        auto expr = parseExpression();
        consume(TOK_CLOSE_PAREN, "Expected ')' after expression");
        return expr;
    } else if (match(TOK_IDENTIFIER)) {
        std::string name = previous().lexeme;

        if (match(TOK_OPEN_PAREN)) {
            // This is a function call
            std::vector<std::unique_ptr<ExprAST>> args;

            if (!check(TOK_CLOSE_PAREN)) {
                do {
                    args.push_back(parseComparison());
                } while (match(TOK_COMMA));
            }

            consume(TOK_CLOSE_PAREN, "Expected ')' after function arguments");

            return std::make_unique<CallExprAST>(name, std::move(args));
        }

        return std::make_unique<VariableExprAST>(name);
    }

    throw std::runtime_error("Expected primary expression");
}

std::string Parser::parseType() {
    if (match(TOK_OPEN_BRACKET)) {
        consume(TOK_CLOSE_BRACKET, "Expected ']' after '['");
        std::string elementType = parseType();
        return "[]" + elementType;
    } else if (match(TOK_TYPE)) {
        return previous().lexeme;
    } else {
        throw std::runtime_error("Expected type");
    }
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
    if (match(TOK_RETURN)) {
        auto expr = parseComparison();
        consume(TOK_SEMI, "Expected ';' after return statement");
        return std::make_unique<ReturnExprAST>(std::move(expr));
    } else if (match(TOK_CONST) || match(TOK_VAR)) {
        bool isConst = previous().type == TOK_CONST;
        consume(TOK_IDENTIFIER, "Expected variable name");
        std::string name = previous().lexeme;

        // Optional type annotation
        std::string type = "u8"; // Default type
        if (match(TOK_COLON)) {
            type = parseType();
        }

        std::unique_ptr<ExprAST> init = nullptr;
        if (match(TOK_EQUAL)) {
            init = parseComparison();
        }
        consume(TOK_SEMI, "Expected ';' after variable declaration");

        return std::make_unique<VarDeclAST>(name, type, isConst, std::move(init));
    } else if (match(TOK_IF)) {
        consume(TOK_OPEN_PAREN, "Expected '(' after 'if'");
        auto condition = parseComparison();
        consume(TOK_CLOSE_PAREN, "Expected ')' after if condition");
        
        consume(TOK_OPEN_BRACE, "Expected '{' after if condition");
        std::vector<std::unique_ptr<ExprAST>> thenBody;
        while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
            thenBody.push_back(parseExpression());
        }
        consume(TOK_CLOSE_BRACE, "Expected '}' after if body");
        
        std::vector<std::unique_ptr<ExprAST>> elseBody;
        if (match(TOK_ELSE)) {
            consume(TOK_OPEN_BRACE, "Expected '{' after 'else'");
            while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
                elseBody.push_back(parseExpression());
            }
            consume(TOK_CLOSE_BRACE, "Expected '}' after else body");
        }
        
        return std::make_unique<IfExprAST>(std::move(condition), std::move(thenBody), std::move(elseBody));
    } else if (match(TOK_WHILE)) {
        consume(TOK_OPEN_PAREN, "Expected '(' after 'while'");
        auto condition = parseComparison();
        consume(TOK_CLOSE_PAREN, "Expected ')' after while condition");
        
        consume(TOK_OPEN_BRACE, "Expected '{' after while condition");
        std::vector<std::unique_ptr<ExprAST>> body;
        while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
            body.push_back(parseExpression());
        }
        consume(TOK_CLOSE_BRACE, "Expected '}' after while body");
        
        return std::make_unique<WhileExprAST>(std::move(condition), std::move(body));
    } else if (match(TOK_FOR)) {
        consume(TOK_IDENTIFIER, "Expected variable name after 'for'");
        std::string varName = previous().lexeme;
        
        consume(TOK_IN, "Expected 'in' after for variable");
        auto start = parseComparison();
        consume(TOK_COLON, "Expected ':' in for range");
        auto end = parseComparison();
        
        consume(TOK_OPEN_BRACE, "Expected '{' after for range");
        std::vector<std::unique_ptr<ExprAST>> body;
        while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
            body.push_back(parseExpression());
        }
        consume(TOK_CLOSE_BRACE, "Expected '}' after for body");
        
        return std::make_unique<ForExprAST>(varName, std::move(start), std::move(end), std::move(body));
    } else if (match(TOK_BREAK)) {
        consume(TOK_SEMI, "Expected ';' after break");
        return std::make_unique<BreakExprAST>();
    } else if (match(TOK_CONTINUE)) {
        consume(TOK_SEMI, "Expected ';' after continue");
        return std::make_unique<ContinueExprAST>();
    } else if (check(TOK_IDENTIFIER)) {
        // Look ahead to see if this is a function call statement
        int saved_current = current;
        advance(); // consume identifier
        if (check(TOK_OPEN_PAREN)) {
            // This is a function call, reset and parse it
            current = saved_current;
            auto expr = parseComparison();
            consume(TOK_SEMI, "Expected ';' after function call");
            return expr;
        } else {
            // Not a function call, reset and continue with normal parsing
            current = saved_current;
        }
    }

    return parseComparison();
}

std::unique_ptr<ExprAST> Parser::parseComparison() {
    auto LHS = parseAddition();

    if (match(TOK_EQUAL_EQUAL)) {
        auto RHS = parseAddition();
        return std::make_unique<BinaryExprAST>("==", std::move(LHS), std::move(RHS));
    } else if (match(TOK_NOT_EQUAL)) {
        auto RHS = parseAddition();
        return std::make_unique<BinaryExprAST>("!=", std::move(LHS), std::move(RHS));
    } else if (match(TOK_LESS)) {
        auto RHS = parseAddition();
        return std::make_unique<BinaryExprAST>("<", std::move(LHS), std::move(RHS));
    } else if (match(TOK_LESS_EQUAL)) {
        auto RHS = parseAddition();
        return std::make_unique<BinaryExprAST>("<=", std::move(LHS), std::move(RHS));
    } else if (match(TOK_GREATER)) {
        auto RHS = parseAddition();
        return std::make_unique<BinaryExprAST>(">", std::move(LHS), std::move(RHS));
    } else if (match(TOK_GREATER_EQUAL)) {
        auto RHS = parseAddition();
        return std::make_unique<BinaryExprAST>(">=", std::move(LHS), std::move(RHS));
    }

    return LHS;
}

std::unique_ptr<ExprAST> Parser::parseAddition() {
    auto LHS = parsePrimary();

    if (match(TOK_PLUS)) {
        auto RHS = parsePrimary();
        return std::make_unique<BinaryExprAST>("+", std::move(LHS), std::move(RHS));
    }

    return LHS;
}

std::unique_ptr<FunctionAST> Parser::parseFunction() {
    // Check for extern or export keywords
    bool isExtern = false;
    bool isExport = false;
    
    if (match(TOK_EXTERN)) {
        isExtern = true;
    } else if (match(TOK_EXPORT)) {
        isExport = true;
    }
    
    consume(TOK_FN, "Expected 'fn' keyword");
    consume(TOK_IDENTIFIER, "Expected function name");
    std::string name = previous().lexeme;

    consume(TOK_OPEN_PAREN, "Expected '(' after function name");

    std::vector<std::pair<std::string, std::string>> args;
    if (!check(TOK_CLOSE_PAREN)) {
        do {
            consume(TOK_IDENTIFIER, "Expected parameter name");
            std::string paramName = previous().lexeme;

            consume(TOK_COLON, "Expected ':' after parameter name");
            std::string paramType = parseType();

            args.emplace_back(paramName, paramType);
        } while (match(TOK_COMMA));
    }

    consume(TOK_CLOSE_PAREN, "Expected ')' after parameters");

    // Parse the return type
    std::string returnType;
    if (match(TOK_ARROW)) {
        returnType = parseType();
    }

    // Extern functions don't have a body
    if (isExtern) {
        consume(TOK_SEMI, "Expected ';' after extern function declaration");
        std::vector<std::unique_ptr<ExprAST>> emptyBody;
        return std::make_unique<FunctionAST>(name, std::move(args), returnType, std::move(emptyBody), true, false);
    }

    consume(TOK_OPEN_BRACE, "Expected '{' before function body");

    std::vector<std::unique_ptr<ExprAST>> body;
    while (!check(TOK_CLOSE_BRACE) && !isAtEnd()) {
        body.push_back(parseExpression());
    }

    consume(TOK_CLOSE_BRACE, "Expected '}' after function body");

    return std::make_unique<FunctionAST>(name, std::move(args), returnType, std::move(body), false, isExport);
}

std::vector<std::unique_ptr<FunctionAST>> Parser::parse() {
    std::vector<std::unique_ptr<FunctionAST>> functions;

    while (!isAtEnd()) {
        functions.push_back(parseFunction());
    }

    return functions;
}
