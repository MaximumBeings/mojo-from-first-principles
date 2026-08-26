# Chapter 7: Backward Function Implementation

Reverse-mode automatic differentiation separates a global problem into local ones. Each node receives the derivative of the loss with respect to its output, multiplies by its own local derivatives, and returns one contribution for each input. This chapter derives those rules numerically before encoding them.

## 7.1 The chain rule is message passing

For `w=x*y+x` at `x=3`, `y=4`, the value `x` reaches `w` through two paths. Through multiplication it contributes `y×1=4`; through addition it contributes 1. Reverse mode sends those two messages backward and adds them at value ID 0.

```mojo
@fieldwise_init
struct InputGrads(Copyable, Movable):
    var left: Float32
    var right: Float32

def local_backward(node: Node, tape: Tape, grad_output: Float32) -> InputGrads:
    if node.op == OpKind.ADD:
        return InputGrads(grad_output, grad_output)
    if node.op == OpKind.MUL:
        return InputGrads(
            grad_output * tape.values[node.right],
            grad_output * tape.values[node.left],
        )
    debug_assert(False, "unknown operation")
    return InputGrads(0, 0)
```

**Manual worked example.** The final ADD receives `grad_output=1`; it returns `(1,1)`, one contribution for `z` and one for `x`. The earlier MUL then receives `z.grad=1`; with stored inputs 3 and 4 it returns `(1×4,1×3)=(4,3)`. Accumulating by value ID yields `x.grad=1+4=5` and `y.grad=3`.

## 7.2 Element-wise tensor rules preserve shape

The scalar arithmetic lifts element by element to tensors. Addition passes the upstream gradient to both operands; multiplication scales it by the opposite operand. Broadcasting adds one extra requirement: reduce a broadcast operand's gradient back to its original shape.

```mojo
def add_backward(grad_out: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    return (
        sum_to_shape(grad_out, a.shape),
        sum_to_shape(grad_out, b.shape),
    )

def mul_backward(grad_out: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    return (
        sum_to_shape(grad_out * b, a.shape),
        sum_to_shape(grad_out * a, b.shape),
    )
```

**Manual worked example.** Let `a=[[1,2,3],[4,5,6]]` and broadcast `b=[10,20,30]` across two rows. If `grad_out` is all ones, `dL/da` is a 2×3 matrix of ones. For `b`, the two broadcast uses add columnwise: `[1+1,1+1,1+1]=[2,2,2]`. Returning `grad_out` unchanged for `b` would have the wrong shape and miss that accumulation.

## 7.3 Matrix multiplication uses transposed contractions

For `Y=A@B`, perturbations obey `dY=dA@B + A@dB`. Matching each term against the upstream matrix `G=dL/dY` gives `dL/dA=G@Bᵀ` and `dL/dB=Aᵀ@G`. Shape checks are part of the derivation, not an afterthought.

```mojo
def matmul_backward(grad_out: Tensor, a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    return (matmul(grad_out, transpose(b)), matmul(transpose(a), grad_out))
```

**Manual worked example.** Use `A=[[1,2,3],[4,5,6]]`, `B=[[1,2],[3,4],[5,6]]`, and `G=[[1,1],[1,1]]`. Then `G@Bᵀ=[[3,7,11],[3,7,11]]`, matching A's 2×3 shape. Also `Aᵀ@G=[[5,5],[7,7],[9,9]]`, matching B's 3×2 shape. Nudging `A[0,0]` changes the first-row outputs by B's first row `[1,2]`, so their summed loss changes at rate `1+2=3`, exactly `dA[0,0]`.

## 7.4 Custom operations need an explicit mathematical contract

A numerical solver may contain many iterations that are poor graph nodes. Treat it as one operation only when its backward rule is derived from the equation the solver satisfies and its domain assumptions are checked.

```mojo
def sqrt_solve_backward(solution: Float32, grad_output: Float32) -> Float32:
    debug_assert(solution > 0, "sqrt derivative requires a positive solution")
    return grad_output / (2 * solution)
```

**Manual worked example.** Bisection solves `x²-c=0` for `c=2`, giving `x≈1.41421356`. Implicit differentiation gives `dx/dc=1/(2x)≈0.353553`. A centered check with `h=0.001` gives `(sqrt(2.001)-sqrt(1.999))/0.002≈0.353553`, agreeing to six decimals. Centering matters: the previous one-sided, rounded check exaggerated the error.

These rules are pure local mathematics. Chapter 8 supplies the global machinery that chooses the right rule, walks nodes in reverse, and sums contributions into the unique gradient slot for each value ID.
