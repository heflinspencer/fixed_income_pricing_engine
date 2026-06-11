import scipy.optimize as optimize

def calculate_bond_price(face_value, coupon_rate, market_yield, years_to_maturity, frequency=2):
    """
    Calculates the current price of a standard coupon bond.
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    total_periods = int(years_to_maturity * frequency)
    rate_per_period = market_yield / frequency
    
    # Present value of coupon payments
    coupon_pv = sum(coupon_payment / (1 + rate_per_period)**t for t in range(1, total_periods + 1))
    
    # Present value of face value (principal)
    face_value_pv = face_value / (1 + rate_per_period)**total_periods
    
    return coupon_pv + face_value_pv

def calculate_ytm(bond_price, face_value, coupon_rate, years_to_maturity, frequency=2):
    """
    Calculates the Yield to Maturity (YTM) of a bond using numerical root-finding.
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    total_periods = int(years_to_maturity * frequency)
    
    # Defining the objective function: Price(y) - Market Price = 0
    def bond_value_objective(y):
        rate_per_period = y / frequency
        coupon_pv = sum(coupon_payment / (1 + rate_per_period)**t for t in range(1, total_periods + 1))
        face_value_pv = face_value / (1 + rate_per_period)**total_periods
        return (coupon_pv + face_value_pv) - bond_price

    # Use Brent's method to find the root (YTM) between 0.001% and 100%
    try:
        ytm_per_period = optimize.brentq(bond_value_objective, 0.00001, 1.0)
        return ytm_per_period * frequency
    except ValueError:
        return None # Returns None if no solution is found in the range

def calculate_duration_and_convexity(face_value, coupon_rate, market_yield, years_to_maturity, frequency=2):
    """
    Calculates Macaulay Duration, Modified Duration, and Convexity.
    """
    coupon_payment = (coupon_rate * face_value) / frequency
    total_periods = int(years_to_maturity * frequency)
    rate_per_period = market_yield / frequency
    bond_price = calculate_bond_price(face_value, coupon_rate, market_yield, years_to_maturity, frequency)
    
    weighted_time_sum = 0
    convexity_sum = 0
    
    for t in range(1, total_periods + 1):
        # Determine cash flow for the period
        cash_flow = coupon_payment
        if t == total_periods:
            cash_flow += face_value
            
        pv_cf = cash_flow / (1 + rate_per_period)**t
        
        # Elements for duration and convexity formulas
        weighted_time_sum += (t / frequency) * pv_cf
        convexity_sum += pv_cf * (t / frequency) * ((t + 1) / frequency)

    macaulay_duration = weighted_time_sum / bond_price
    modified_duration = macaulay_duration / (1 + rate_per_period)
    convexity = convexity_sum / (bond_price * (1 + rate_per_period)**2)
    
    return macaulay_duration, modified_duration, convexity

# --- Example Usage ---
if __name__ == "__main__":
    # Bond Characteristics
    FACE_VALUE = 1000
    COUPON_RATE = 0.05  # 5% annual coupon
    YIELD_TO_MATURITY = 0.06  # 6% market yield
    YEARS = 10
    FREQUENCY = 2  # Semi-annual payments

    print("--- Fixed Income Valuation Report ---")
    
    # 1. Price the bond
    price = calculate_bond_price(FACE_VALUE, COUPON_RATE, YIELD_TO_MATURITY, YEARS, FREQUENCY)
    print(f"Theoretical Bond Price: ${price:.2f}")
    
    # 2. Back-calculate YTM using the price we just found (to verify accuracy)
    calculated_ytm = calculate_ytm(price, FACE_VALUE, COUPON_RATE, YEARS, FREQUENCY)
    print(f"Calculated YTM:         {calculated_ytm * 100:.2f}%")
    
    # 3. Risk Metrics (Duration & Convexity)
    mac_dur, mod_dur, conv = calculate_duration_and_convexity(FACE_VALUE, COUPON_RATE, YIELD_TO_MATURITY, YEARS, FREQUENCY)
    print(f"Macaulay Duration:      {mac_dur:.3f} years")
    print(f"Modified Duration:      {mod_dur:.3f} years")
    print(f"Convexity:              {conv:.3f}")