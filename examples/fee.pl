% A call to another predicate, which a TRITON kernel has to INLINE to compile at all.
%
% A kernel is a flat block of lanes with no call stack, so nothing here can be CALLED in the
% ordinary sense. A call that cannot be re-entered does not need a stack, though -- it needs
% the callee pasted into the caller -- so the compiler inlines fee/3 at both of the sites
% below, and goes on refusing only what no amount of pasting terminates, which is recursion.
%
% Three things about fee/3 are here to be got wrong, because each one produces a kernel that
% compiles and answers anyway:
%
%   ITS CLAUSES OVERLAP. A small amount matches the flat-minimum clause AND the general one,
%   and they disagree -- 50 at a divisor of 2 is a fee of 10, not 25. Program order is the
%   whole of what decides it, and an inlined frame has to enforce that exactly as a
%   top-level predicate does.
%
%   IT IS PARTIAL. A divisor of zero gives no answer at all, so an amount of 100 or more
%   fails. That failure has to reach total/4: the frame leaves a value behind either way.
%
%   IT NAMES THE CALLER'S VARIABLES. fee/3 writes A and R, which total/4 also writes, and
%   the FIRST call passes B -- so an expansion that did not rename would assign the caller's
%   A from B and charge B twice.
%
% The sum can overflow as well, at which point the whole predicate fails rather than
% reporting the wrapped total.
%
% Called (+In, +In, +In, -Out): total(A, B, R, T), the fees on two amounts added.
fee(A, R, F) :-
    A < 100,
    F is 10.
fee(A, R, F) :-
    F is A // R.

total(A, B, R, T) :-
    fee(B, R, FB),
    fee(A, R, FA),
    T is FA + FB.
