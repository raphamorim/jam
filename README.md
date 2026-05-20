<!-- LOGO -->
<h1>
<p align="center">
  <img src="./docs/assets/images/mascot-01.png" alt="Jam Logo" width="256">
  <br>Jam Programming Language
</h1>
  <p align="center">
    <a href="https://jamlang.org">Reference</a>
    ·
    <a href="https://github.com/sponsors/raphamorim">Sponsor</a>
  </p>
</p>

More about the language [here](https://rapha.land/jam-programming-language/).

There's nothing available to use yet, and I've also disabled issues and pull requests on the repo. I'll re-enable both after the v0.1 release and once there's a proper documentation website. If there's anything you'd like to ask directly, feel free to send me an email.

Jam will move to an org in the future, but right now it's mostly me maintaining the language and I'm considering testing other options besides GitHub, so I'd like to wait a bit before starting to move things around.

In case you want to test before v0.1:

```bash
brew tap raphamorim/jam https://github.com/raphamorim/jam
brew install --HEAD jam
```

Quick overview:

```jam
const std = import("std");
const {Vec} = import("collections");

const Histogram = struct {
    bins: Vec(u32),

    // `cfn` opts a method into the compiler's synthesized-call set.
    //
    //   cfn default()       opt-in constructor
    //   cfn drop(...)       auto-fires at every scope exit
    //   cfn at(self, i)     desugars `obj[i]`     (read)
    //   cfn setAt(...)      desugars `obj[i] = x` (write)
    cfn default() Self {
        var bins: Vec(u32) = Vec(u32).withCapacity(256);
        var i: u32 = 0;
        while (i < 256) {
            bins.push(0);
            i = i + 1;
        }
        return Self { bins: bins };
    }
    cfn drop(self: mut Self) {
        std.fmt.println("drop has happened");
    }
    cfn at(self: Self, i: u32) u32 {
        return self.bins[i];
    }
    cfn setAt(self: mut Self, i: u32, value: u32) {
        self.bins[i] = value;
    }

    fn record(self: mut Self, byte: u8) {
        const i: u32 = byte as u32;
        self[i] = self[i] + 1;
    }
};

fn main() {
    var h: Histogram = Histogram.default();
    h.record(7);
    h.record(7);
    h.record(42);
    std.fmt.println(h[7]);    // 2, via h.at(7)
    std.fmt.println(h[42]);   // 1
}
```

## Contributing

1. [Donate monthly](https://github.com/sponsors/raphamorim).

2. No AI Policy (issues and pull requests). You can comment in your native language if you are not confortable with English.

## References

- Chris Lattner et al., "MLIR: A Compiler Infrastructure for the End of Moore's Law", 2020. https://arxiv.org/abs/2002.11054
- Dimi Racordon, Dave Abrahams, et al., “Implementation Strategies for Mutable Value Semantics”, Journal of Object Technology, 2022. https://research.google/pubs/mutable-value-semantics/
- Luc Maranget, “Compiling Pattern Matching to Good Decision Trees”, ACM SIGPLAN Workshop on ML, 2008. http://moscova.inria.fr/~maranget/papers/ml05e-maranget.pdf
- Zig v0.10 — referenced for its flat, tag-dispatched AST, type interning, and Debug-by-default codegen approach. https://github.com/ziglang/zig/tree/0.10.x
- The Rust compiler (rustc) — referenced for its `-C key=value` codegen-options CLI shape (opt-level / lto / strip) and ABI by-value-vs-by-pointer threshold heuristics. https://github.com/rust-lang/rust
- Swift's mutable value semantics — the `var`/`let` distinction and law-of-exclusivity model that informed Jam's `let` / `mut` / `move` binding modes and aliasing rules. See Racordon & Abrahams (cited above) for the formal treatment; Dave Abrahams is the principal designer of Swift's value semantics, and the paper uses Val/Hylo as a research vehicle for the underlying MVS ideas. https://www.hylo-lang.org/

## Artwork

Jam logo/mascot was created by [Anthony Orozco](https://www.behance.net/ntnay).

## License

This software is distributed under the Apache License 2.0 with LLVM Exceptions.
