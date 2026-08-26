# 0.3 Memory Layout Strategies

Memory layout decides which values share cache lines and GPU memory transactions. Array-of-structs (AoS) keeps each record together; structure-of-arrays (SoA) keeps each field together. Choose from the access pattern, not from aesthetics.

## 0.3.1 AoS favors whole-record access

AoS is natural when each iteration consumes most fields of one object.

```mojo
@fieldwise_init
struct Bond(Copyable, Movable):
    var face: Float32
    var rate: Float32
    var years: Float32

def simple_interest(bond: Bond) -> Float32:
    return bond.face * bond.rate * bond.years
```

**Manual worked example.** A bond `(face=100, rate=0.05, years=2)` produces `100×0.05×2=10`. All three neighboring fields are consumed together, which suits AoS.

## 0.3.2 SoA favors one-field sweeps

SoA stores each field in a contiguous list. A vectorized or GPU loop over rates then reads neighboring values with no unrelated fields between them.

```mojo
@fieldwise_init
struct BondBook:
    var face: List[Float32]
    var rate: List[Float32]
    var years: List[Float32]
```

**Manual worked example.** For three bonds, rates `[0.04,0.05,0.06]` occupy 12 consecutive payload bytes. Threads 0–2 read adjacent rates; in AoS, each read would skip the face and years fields of neighboring records.

## 0.3.3 Layout conversion has a real cost

Converting AoS to SoA reads every record and writes every field. Do it once at a boundary when many later kernels benefit, not inside each operation.

```mojo
def extract_rates(bonds: List[Bond]) -> List[Float32]:
    var rates = List[Float32](capacity=len(bonds))
    for bond in bonds:
        rates.append(bond.rate)
    return rates
```

**Manual worked example.** Three bonds with rates 4%, 5%, and 6% require three reads and three writes to produce `[0.04,0.05,0.06]`. Repeating that conversion before ten kernels would perform 30 redundant field copies.

## 0.3.4 Tensor metadata is naturally SoA

A tensor separates data, shape, and strides because kernels usually sweep data while reading the small metadata once. Mixing shape fields into every element would multiply metadata storage by element count.

```text
Tensor = data buffer + shape vector + stride vector + device descriptor
```

**Manual worked example.** A 1,000×1,000 Float32 tensor has one million data values but only two shape values `[1000,1000]` and two strides `[1000,1]`. Repeating those four metadata integers beside every scalar would add millions of unnecessary fields.

AoS and SoA are complementary tools: whole-record logic prefers AoS, while tensor kernels and fieldwise analytics usually prefer SoA.
