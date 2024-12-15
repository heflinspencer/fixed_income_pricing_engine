import numpy as np
from scipy import optimize

class FixedIncomePricingEngine:
    def __init__(self, face_value, coupon_rate, frequency, maturity, yield_curve):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.frequency = frequency
        self.maturity = maturity
        self.yield_curve = yield_curve

    def price_bond(self):
        coupon = self.face_value * self.coupon_rate / self.frequency
        periods = np.arange(1, self.maturity * self.frequency + 1) / self.frequency
        cash_flows = np.array([coupon] * len(periods))
        cash_flows[-1] += self.face_value
        discount_factors = np.exp(-self.yield_curve(periods) * periods)
        return np.sum(cash_flows * discount_factors)

    def calculate_yield_to_maturity(self, price):
        def objective(ytm):
            return self.price_bond() - price
        return optimize.newton(objective, x0=0.05)  # Use 5% as initial guess
