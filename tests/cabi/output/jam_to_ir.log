ld: open() failed, errno=21 for 'output'
clang: error: linker command failed with exit code 1 (use -v to see invocation)
; ModuleID = 'my cool compiler'
source_filename = "my cool compiler"

@str = private constant [43 x i8] c"Calling C function add_numbers from Jam...\00"

define i8 @main() {
entry:
  %puts_call = call i32 @puts(ptr @str)
  ret i8 0
}

declare i32 @printf(ptr, ...)

declare i32 @puts(ptr)
Compilation completed successfully.
