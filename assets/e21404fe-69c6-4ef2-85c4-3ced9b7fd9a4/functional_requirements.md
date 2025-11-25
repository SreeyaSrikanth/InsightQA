# Checkout Page – Functional Requirements

## 1. Product Display
- The system must display at least one product with:
  - Product name
  - Product description or size
  - Price in USD
- Users must be able to add items to the cart using a button.

## 2. Cart Behaviour
- The cart must show the current quantity of each item.
- If no items exist, the cart must show “No items yet.”
- Quantity increments by 1 each time the "Add to cart" button is clicked.
- Quantity cannot become negative.

## 3. Price Calculations
- Subtotal = Item price × Quantity
- Discount is applied to the subtotal.
- Total = Subtotal – Discount
- Total cannot go below 0.

## 4. Discount Application
- Users can enter a discount code and click “Apply.”
- The system must show:
  - Discount applied (in USD)
  - Updated total price

## 5. Error Handling
- If invalid discount code is used, discount must be set to 0.
- If user tries to make a payment with:
  - empty cart
  - total = 0  
  system must show an error message.

## 6. Payment Flow
- Clicking “Pay Now” should perform validation.
- If valid:
  - Show “Payment successful!”
- If invalid:
  - Show error “Cannot pay: cart is empty or total is zero.”

## 7. UI Elements
- All IDs should be unique.
- All interactive components must be accessible via DOM selectors.
