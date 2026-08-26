# Chapter 8: Gradient Computation Engine

The backward engine combines three ideas already established: seed the selected scalar output with 1, visit recorded nodes in reverse topological order, and add every local contribution into the corresponding input's gradient slot. The tape owns those slots, so aliases and repeated uses cannot split a gradient across copied objects.

## 8.1 One gradient slot per value

Gradients are indexed exactly like values. Extending the tape with a parallel list makes the invariant mechanical: `values[i]` and `grads[i]` always describe the same differentiable quantity.

```mojo
# Add this field to Tape:
#     var grads: List[Float32]

def append_value(mut tape: Tape, value: Float32, requires_grad: Bool) -> ValueId:
    var result = len(tape.values)
    tape.values.append(value)
    tape.requires_grad.append(requires_grad)
    tape.grads.append(0)
    return result
```

**Manual worked example.** After appending `x=3`, `y=4`, `z=12`, and `w=15`, the three parallel lists have length four and `grads=[0,0,0,0]`. Value ID 0 means `x` everywhere. There is no separate `node.grad`, so seeding or accumulating a gradient cannot update one object while backward reads another.

## 8.2 Reverse-mode execution

Backward must reject a non-scalar output unless the caller supplies an explicit vector-Jacobian seed. For the scalar path taught here, `dL/dL=1` is the unique natural seed.

```mojo
def backward(mut tape: Tape, loss: ValueId):
    for i in range(len(tape.grads)):
        tape.grads[i] = 0
    tape.grads[loss] = 1

    var node_index = len(tape.nodes)
    while node_index > 0:
        node_index -= 1
        ref node = tape.nodes[node_index]
        var upstream = tape.grads[node.output]
        if upstream == 0:
            continue
        var local = local_backward(node, tape, upstream)
        if tape.requires_grad[node.left]:
            tape.grads[node.left] += local.left
        if tape.requires_grad[node.right]:
            tape.grads[node.right] += local.right
```

**Manual worked example.** Seed `grads[3]=1`. Visit ADD `(2,0→3)`: add 1 to slots 2 and 0, producing `[1,0,1,1]`. Visit MUL `(0,1→2)`: its upstream is slot 2's 1, so add `1×4=4` to slot 0 and `1×3=3` to slot 1. The final list is `[5,3,1,1]`; the leaf answers are `dx=5`, `dy=3`.

## 8.3 Accumulation is required for branches

Assignment is wrong when a value is used more than once. The `+=` operations above are the implementation of the multivariable chain rule's sum over paths.

```mojo
def accumulate(mut tape: Tape, value: ValueId, contribution: Float32):
    if tape.requires_grad[value]:
        tape.grads[value] += contribution
```

**Manual worked example.** The ADD node contributes 1 to `x`; the MUL node later contributes 4 to the same ID. After the first call, slot 0 is 1. After the second, it is 5. Replacing `+=` with `=` would leave 4 and fail the centered finite-difference check `(w(3.001)-w(2.999))/0.002=5`.

## 8.4 Lifetime and memory policy

An eager tape is single-use by default: clear its nodes after backward unless the caller explicitly requests retention. Tensor implementations should save only values required by a local derivative, but they must never discard the IDs needed to route gradients.

```mojo
def release_graph(mut tape: Tape, retain_graph: Bool):
    if not retain_graph:
        tape.nodes.clear()
```

**Manual worked example.** A forward pass with two nodes retains two node records through backward. Calling `release_graph(tape, False)` changes the node count from 2 to 0 while leaving the four computed values and gradients available for inspection. Calling backward again now has no path to the leaves, which is why a second backward requires a new forward pass or `retain_graph=True`.

## 8.5 Verification before optimization

Every new local rule should pass both an analytic spot check and a centered finite-difference check before receiving a SIMD or GPU implementation. Use a relative error denominator that remains stable near zero.

```mojo
def relative_error(analytic: Float64, numeric: Float64) -> Float64:
    return abs(analytic - numeric) / max(1.0, max(abs(analytic), abs(numeric)))
```

**Manual worked example.** For `w=x*y+x`, analytic `dx=5`. With `h=0.001`, the numeric derivative is `(15.005-14.995)/0.002=5`. The numerator is 0 and the denominator is 5, so relative error is 0. For a near-zero derivative such as `1e-9` versus 0, the denominator clamps to 1 and reports the meaningful absolute-scale error `1e-9` rather than exploding.

The scalar tape now works end to end without the aliasing bug of a separate node gradient. A tensor engine replaces `Float32` values and gradients with graph-owned tensor buffers, but keeps the same value IDs, reverse traversal, accumulation rule, and verification discipline.
