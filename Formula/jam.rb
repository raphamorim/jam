# Homebrew formula for the Jam compiler.
#
# Installation (HEAD-only until the first tagged release):
#
#   brew tap raphamorim/jam https://github.com/raphamorim/jam
#   brew install --HEAD jam
#
# Once a release is cut, add a `stable do ... end` block with the
# tarball URL + sha256, and users can drop `--HEAD`.

class Jam < Formula
  desc "Jam programming language"
  homepage "https://github.com/raphamorim/jam"
  license "Apache-2.0" => { with: "LLVM-exception" }
  head "https://github.com/raphamorim/jam.git", branch: "main"

  depends_on "cmake" => :build
  depends_on "llvm@22"

  def install
    llvm = Formula["llvm@22"]
    ENV.prepend_path "PATH", llvm.opt_bin

    system "cmake", "-S", ".", "-B", "build",
                    "-DCMAKE_PREFIX_PATH=#{llvm.opt_prefix}",
                    *std_cmake_args
    system "cmake", "--build", "build"
    system "cmake", "--install", "build"
  end

  test do
    # Verify the std lookup works end-to-end: the installed `jam` walks
    # up from `#{prefix}/bin` and finds `#{prefix}/lib/jam/std/`, so
    # `import("std/collections")` resolves without any flag.
    (testpath/"hello.jam").write <<~JAM
      const { Vec } = import("std/collections");
      const { Option } = import("std/option");
      fn main() {
        var v: Vec(i32) = Vec(i32).empty();
        v.push(42);
      }
    JAM
    system bin/"jam", testpath/"hello.jam"
    assert_path_exists testpath/"output"
  end
end
