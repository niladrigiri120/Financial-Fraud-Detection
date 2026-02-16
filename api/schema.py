from pydantic import BaseModel
from typing import Optional

class Transaction(BaseModel):
    type: Optional[str] = None

    amount: float

    oldbalanceOrg: float

    newbalanceOrig: Optional[float] = None

    tx_count_24h: Optional[int] = 0
    avg_amount_24h: Optional[float] = None
