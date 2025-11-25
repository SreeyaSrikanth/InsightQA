# Payment Flow Specification

## 1. Preconditions
- User must have at least one item in the cart.
- Total price must be greater than 0.

## 2. Main Flow
1. User clicks the “Pay Now” button.
2. System validates:
   - quantity > 0
   - total price > 0
3. If validation fails:
   - Show an error message in red.
4. If validation passes:
   - Show a green success message saying “Payment successful!”

## 3. UI Requirements
- Success message must be shown in the element with ID `payment_message`.
- Error message must be shown in the element with ID `error_message`.

## 4. Testability Notes
- No actual payment gateway is called.
- All logic is handled client-side.
