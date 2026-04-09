"""Variables to replace numeric literals with semantically meaningful identifiers.

These are from the semiotics module in `hunterMakesPy`. https://github.com/hunterhogan/hunterMakesPy
"""
decreasing: int = -1
"""Express descending iteration or a reverse direction.

The identifier `decreasing` holds the value `-1` and serves as a semantic replacement for
numeric literals in contexts where direction or ordering matters. You can use `decreasing`
as an addend to adjust boundary values, as a multiplicand to reverse sign or direction,
or in both roles simultaneously to express complex transformations.

You can use `decreasing` wherever `-1` would appear but where the meaning "descending order"
or "reverse direction" is more important than the specific numeric value. Using `decreasing`
makes the code's intent explicit and communicates the semantic purpose to readers who might
not immediately recognize `-1` as a directional indicator.

Common contexts include: reverse iteration through sequences, computing predecessors or
backward offsets, negating dimensions or indices, and constructing `range` objects that
count downward.

Examples
--------
You can use `decreasing` as an addend to compute a loop boundary:

>>> for countDown in range(dimension - 2 + decreasing, decreasing, decreasing):
...     processValue(countDown)

In this example, `dimension - 2 + decreasing` computes the starting index (equivalent to
`dimension - 3`), the second `decreasing` stops iteration at `-1` (to include `0`), and
the third `decreasing` specifies the step direction (counting down by `1`).

You can use `decreasing` as a multiplicand to reverse sign:

>>> negatedOffset = baseOffset * decreasing

You can use `decreasing` as both multiplicand and addend in a single expression:

>>> adjustedIndex = dimensionHeadSecond * decreasing + decreasing

This pattern appears when converting between coordinate systems or computing reverse-indexed
positions with additional boundary adjustments.

You can use `decreasing` as the step parameter in `range` [1] to iterate backward:

>>> for leaf in range(leavesTotal + decreasing, 1, decreasing):
...     processLeaf(leaf)

You can combine `decreasing` with other semantic constants:

>>> boundaryRange = range(
...     (start + inclusive) * decreasing,
...     (stop + inclusive) * decreasing,
...     decreasing
... )

References
----------
[1] Built-in Functions - `range` (Python documentation)
	https://docs.python.org/3/library/functions.html#func-range

"""

zeroIndexed: int = 1
"""Express that the adjustment to a value is due to zero-based indexing.

The identifier `zeroIndexed` holds the value `1` and serves as a semantic replacement
for the numeric literal `1` when converting between zero-based indexing (Python's default)
and one-based indexing (common in mathematical notation, human-readable numbering, and
many domain-specific conventions). You can use `zeroIndexed` as an addend or subtrahend
to adjust index values, counts, or boundary computations.

You can use `zeroIndexed` wherever `1` would appear but where the meaning "adjust for
indexing convention" is more important than the specific numeric value. Using `zeroIndexed`
makes the code's intent explicit: the adjustment exists to reconcile indexing systems, not
to perform arbitrary arithmetic.

The most common usage is `count - zeroIndexed` to convert a one-based count ("there are N
items") to the zero-based index of the last item (`N - 1`). You can also use `+ zeroIndexed`
when converting from zero-based indices back to one-based positions or counts, or when
adjusting formulas that assume one-based indexing.

Common contexts include: computing final indices from counts, converting between mathematical
notation (often one-based) and Python code (zero-based), accessing the last valid index of
a sequence, and boundary calculations in algorithms that mix indexing conventions.

Examples
--------
You can use `- zeroIndexed` to convert a count to the last valid zero-based index:

>>> lastDimensionIndex = dimensionsTotal - zeroIndexed

If `dimensionsTotal` is `3` (representing three dimensions), then `lastDimensionIndex`
becomes `2`, which correctly identifies the index of the third dimension in a zero-indexed
array `[0, 1, 2]`.

You can use `- zeroIndexed` when accessing elements by position:

>>> voodooMath: int = creaseAnteAt二Ante首[
...     largestPossibleLengthOfListOfCreases - zeroIndexed
... ]

Here, `largestPossibleLengthOfListOfCreases` represents a count ("how many elements"),
and subtracting `zeroIndexed` produces the index of the last element.

You can use `- zeroIndexed` in `range` [1] boundary computations:

>>> listIndicesCreasePostToKeep.extend(
...     range(
...         dimensionsTotal - dimensionHead + 1,
...         dimensionsTotal - zeroIndexed
...     )
... )

You can use `- zeroIndexed` in conditional expressions:

>>> if dimensionsTotal - zeroIndexed - dimensionHead == zerosAtThe首:
...     applySpecialCase()

You can use `+ zeroIndexed` when the adjustment goes in the opposite direction:

>>> productsOfDimensionsTruncator: int = (
...     dimensionFrom首 - (dimensionsTotal + zeroIndexed)
... )

In this example, `dimensionsTotal + zeroIndexed` adjusts the total count upward before
subtraction, compensating for a formula that expects one-based indexing.

References
----------
[1] Built-in Functions - `range` (Python documentation)
	https://docs.python.org/3/library/functions.html#func-range

"""
