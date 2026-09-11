% The four integer divisions Prolog distinguishes, which is where the TRITON target had to be
% made to agree with the others.
%
% Triton's `//` and `%` TRUNCATE toward zero, the way C does and the way Python does not.
% Prolog's `//` and `rem` truncate too, so those lower directly; `div` and `mod` FLOOR, and
% the two disagree exactly when the signs differ and the division is not exact -- -7 div 2 is
% -4, and truncation says -3. Nothing about the emitted kernel looks wrong when that is got
% wrong, which is the reason this file exists.
%
% B = 0 has no answer at all, and neither has -2^63 divided by -1, so those lanes come back
% reported as failures rather than holding whatever the hardware left behind.
%
% Called (+In, +In, -Out, -Out, -Out, -Out): divs(A, B, Q, F, R, M) with
%   Q is A // B  (truncating),  F is A div B  (flooring),
%   R is A rem B (sign of A),   M is A mod B  (sign of B).
divs(A, B, Q, F, R, M) :-
    Q is A // B,
    F is A div B,
    R is A rem B,
    M is A mod B.
