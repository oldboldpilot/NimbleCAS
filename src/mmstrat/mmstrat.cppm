// NimbleCAS money-market strategies — repo, discount instruments, and deposit strips.
// @author Olumuyiwa Oluwasanmi
//
// SCOPE. The short-end trading toolkit above nimblecas.finance's T-bill / discount
// helpers: repo (and reverse-repo) interest and repayment on an actual/360 basis;
// discount-instrument price <-> bank-discount-rate conversion; the bank-discount ->
// bond-equivalent (investment) yield conversion for like-for-like comparison; simple
// holding-period return; and the effective compounded rate of a DEPOSIT STRIP (a ladder
// of forward money-market deposits), the money-market analogue of a futures/FRA strip.
//
// HONESTY (config/cpp_details.txt Rule 32, AGENTS.md). These are EXACT arithmetic
// identities on `double` day-count conventions — no root-finding, no series. The
// day-count basis (360 vs 365) is an explicit parameter, never hidden, because the
// bank-discount vs bond-equivalent distinction is exactly where money-market quoting bugs
// live. A non-physical input (non-positive face/price/basis, an empty strip, a
// mismatched-length strip) returns an honest `MathError::domain_error`; a zeroed
// denominator returns `division_by_zero`. Nothing throws.

export module nimblecas.mmstrat;

import std;
import nimblecas.core;

export namespace nimblecas::mmstrat {

// Repo interest on `principal` at `repo_rate` for `days` on an actual/`basis` (default
// 360) count: principal·repo_rate·days/basis. A reverse repo is the same cash flow from
// the other side. basis <= 0 or days < 0 -> domain_error.
[[nodiscard]] auto repo_interest(double principal, double repo_rate, double days,
                                 double basis = 360.0) -> Result<double>;

// Repo repayment = principal + repo_interest. Same guards.
[[nodiscard]] auto repo_repayment(double principal, double repo_rate, double days,
                                  double basis = 360.0) -> Result<double>;

// Price of a discount instrument from its bank-discount rate:
//   price = face·(1 − discount_rate·days/basis). face <= 0, basis <= 0, or days < 0 ->
// domain_error.
[[nodiscard]] auto discount_price(double face, double discount_rate, double days,
                                  double basis = 360.0) -> Result<double>;

// The inverse: bank-discount rate implied by a price.
//   discount_rate = (face − price)/face · basis/days. face <= 0, days <= 0, basis <= 0 ->
// domain_error.
[[nodiscard]] auto discount_rate_from_price(double face, double price, double days,
                                            double basis = 360.0) -> Result<double>;

// Bond-equivalent (investment) yield from a bank-discount rate, so a bill can be compared
// to a coupon instrument: BEY = (365·d)/(360 − d·days) for days <= 182 (the standard
// short-bill form). d·days >= 360 (degenerate) -> division_by_zero; bad days -> domain_error.
[[nodiscard]] auto bond_equivalent_yield(double discount_rate, double days) -> Result<double>;

// Simple holding-period return = (end_price − begin_price + income)/begin_price.
// begin_price <= 0 -> domain_error.
[[nodiscard]] auto holding_period_return(double begin_price, double end_price, double income)
    -> Result<double>;

// Effective compounded rate of a deposit strip: forward simple rates `rates[i]` each
// applied over accrual `accruals[i]` (in years), compounded, then annualised over the
// total tenor: (∏(1 + rateᵢ·accrualᵢ))^(1/Σaccrual) − 1. Empty / mismatched-length /
// non-positive total tenor -> domain_error.
[[nodiscard]] auto deposit_strip_effective_rate(std::span<const double> rates,
                                                std::span<const double> accruals) -> Result<double>;

}  // namespace nimblecas::mmstrat

// ===========================================================================
// Implementation.
// ===========================================================================
namespace nimblecas::mmstrat {

auto repo_interest(double principal, double repo_rate, double days, double basis) -> Result<double> {
    if (basis <= 0.0 || days < 0.0) { return make_error<double>(MathError::domain_error); }
    return principal * repo_rate * days / basis;
}

auto repo_repayment(double principal, double repo_rate, double days, double basis) -> Result<double> {
    return repo_interest(principal, repo_rate, days, basis)
        .transform([principal](double i) { return principal + i; });
}

auto discount_price(double face, double discount_rate, double days, double basis) -> Result<double> {
    if (face <= 0.0 || basis <= 0.0 || days < 0.0) { return make_error<double>(MathError::domain_error); }
    return face * (1.0 - discount_rate * days / basis);
}

auto discount_rate_from_price(double face, double price, double days, double basis) -> Result<double> {
    if (face <= 0.0 || days <= 0.0 || basis <= 0.0) { return make_error<double>(MathError::domain_error); }
    return (face - price) / face * (basis / days);
}

auto bond_equivalent_yield(double discount_rate, double days) -> Result<double> {
    if (days <= 0.0) { return make_error<double>(MathError::domain_error); }
    const double denom = 360.0 - discount_rate * days;
    if (denom == 0.0) { return make_error<double>(MathError::division_by_zero); }
    return (365.0 * discount_rate) / denom;
}

auto holding_period_return(double begin_price, double end_price, double income) -> Result<double> {
    if (begin_price <= 0.0) { return make_error<double>(MathError::domain_error); }
    return (end_price - begin_price + income) / begin_price;
}

auto deposit_strip_effective_rate(std::span<const double> rates, std::span<const double> accruals)
    -> Result<double> {
    if (rates.empty() || rates.size() != accruals.size()) {
        return make_error<double>(MathError::domain_error);
    }
    double growth = 1.0;
    double total = 0.0;
    for (std::size_t i = 0; i < rates.size(); ++i) {
        growth *= (1.0 + rates[i] * accruals[i]);
        total += accruals[i];
    }
    if (total <= 0.0) { return make_error<double>(MathError::domain_error); }
    return std::pow(growth, 1.0 / total) - 1.0;
}

}  // namespace nimblecas::mmstrat
