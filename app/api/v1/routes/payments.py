from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
import stripe

from app.core.config import settings

router = APIRouter()

stripe.api_key = settings.stripe_secret_key
STRIPE_PUBLIC_KEY = settings.stripe_public_key

# ============================ #
# 🔹 1️⃣ Endpoint para Crear Sesión de Pago
# ============================ #
class CheckoutRequest(BaseModel):
    product_name: str
    price: float
    quantity: int

print(f"Stripe API key: {stripe.Customer.list()}")

@router.post("/create-checkout-session")
async def create_checkout_session(data: CheckoutRequest):
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            mode="payment",
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": data.product_name},
                        "unit_amount": int(data.price * 100),  # Convertir dólares a centavos
                    },
                    "quantity": data.quantity,
                }
            ],
            success_url="http://localhost:3000/success",
            cancel_url="http://localhost:3000/cancel",
        )
        print(f"Stripe API key: {stripe.Customer.list()}")
        # print(stripe.Customer.retrieve('cs_test_a152n9KaToZ2GSERsq1Z57ghITAvbaA2YXQk9cF4KbnvytnQjTBAip9wDy'),api_key=settings.stripe_secret_key)
        return {"session_id": session.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================ #
# 🔹 2️⃣ Webhook para Confirmar Pagos (Opcional)
# ============================ #
@router.post("/webhook")
async def stripe_webhook(request: Request):
    webhook_secret = settings.stripe_webhook_secret
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="⚠️ Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="⚠️ Invalid signature")

    # 🔹 Manejar diferentes eventos de Stripe
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        print(f"✅ Pago confirmado: {session}")

    return {"message": "Evento recibido correctamente"}


# ============================ #
# 🔹 3️⃣ Endpoint para Obtener Clave Pública (para el frontend)
# ============================ #
@router.get("/stripe-key")
async def get_stripe_key():
    return {"publicKey": STRIPE_PUBLIC_KEY}


