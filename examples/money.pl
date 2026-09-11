% Money, compiled to a 128-bit FIXED-POINT DECIMAL: `dec` for two digits after the point,
% `dec:N` for N of them.
%
%   prolog_compile examples/money.pl with_tax iio cpp dec
%
% VALUES ARE MINOR UNITS. At `dec` the literal 1999 is 19.99 and 300 is 3.00, because the
% reader has no decimal literal and the source is therefore written in the unit the
% representation uses. The scale is stated once, in the generated file.
%
% ARITHMETIC IS EXACT OR REFUSED, never rounded. 19.99 * 3.00 is 59.97 and is answered;
% 0.01 * 0.01 is 0.0001, which two digits cannot hold, and is refused rather than quietly
% returned as zero. `mod` and `rem` are refused outright: their integer meaning does not
% survive scaling, and picking a convention silently is the failure this whole module exists
% to avoid.

% line_total(+UnitPrice, +Quantity, -Total)
line_total(Price, Qty, Total) :-
    Total is Price * Qty.

% with_tax(+Amount, +Rate, -Total). The rate is an amount like any other, so 10% is 0.10,
% which at `dec` is written 10.
with_tax(Amount, Rate, Total) :-
    Tax is Amount * Rate,
    Total is Amount + Tax.

% split(+Amount, +Ways, -Each) — an EXACT split, or no answer at all. 10.00 four ways is 2.50;
% 10.00 three ways is not representable at two digits and fails rather than losing a cent.
split(Amount, Ways, Each) :-
    Each is Amount // Ways.

% settle(+Amount, +Rate, +Ways, -Each) — the whole transaction in one predicate: add the tax,
% then split the total exactly. It is also what exercises a CALL between compiled predicates
% at this width.
settle(Amount, Rate, Ways, Each) :-
    with_tax(Amount, Rate, Total),
    split(Total, Ways, Each).
