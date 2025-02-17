import stripe

from app.core.config import settings

stripe.api_key = settings.stripe_secret_key

print(f"Stripe API key: {stripe.Customer.list()}")