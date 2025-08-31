# server/history_manager.py
import math

class DualEMAHistory:
    """
    Tracks per-client short/long EMA of a scalar anomaly and produces:
      - suspicion: relu((ema_s - ema_l)/(|ema_l| + eps))
      - trust: exp(-beta * suspicion)
      - confidence: 0..1 agreement of EMAs, tempered by maturity
    """
    def __init__(self, num_clients, alpha_short=0.3, alpha_long=0.05, beta=1.0, eps=1e-8):
        self.eps = eps
        self.beta = beta
        self.alpha_s = alpha_short
        self.alpha_l = alpha_long
        self.state = {
            cid: {
                "ema_s": 0.0,
                "ema_l": 0.0,
                "rounds": 0,
                "suspicion": 0.0,
                "trust": 1.0,
                "confidence": 0.5,  # start neutral
                "last_anom": None
            } for cid in range(num_clients)
        }

    def _ema(self, prev, x, alpha):
        return (1 - alpha) * prev + alpha * x

    def update(self, cid, anomaly_value):
        st = self.state[cid]
        st["rounds"] += 1
        x = float(anomaly_value)

        if st["rounds"] == 1:
            st["ema_s"] = x
            st["ema_l"] = x
        else:
            st["ema_s"] = self._ema(st["ema_s"], x, self.alpha_s)
            st["ema_l"] = self._ema(st["ema_l"], x, self.alpha_l)

        delta = st["ema_s"] - st["ema_l"]
        denom = abs(st["ema_l"]) + self.eps
        delta_norm = delta / denom
        suspicion = max(0.0, delta_norm)
        trust = math.exp(-self.beta * suspicion)

        # agreement of EMAs (stable => higher), blended with maturity
        agree = 1.0 - min(1.0, abs(delta) / (abs(st["ema_s"]) + abs(st["ema_l"]) + self.eps))
        maturity = 1.0 - math.exp(-st["rounds"] / 10.0)  # rises ~over 10 rounds
        confidence = max(0.0, min(1.0, 0.5 * agree + 0.5 * maturity))

        st["suspicion"] = suspicion
        st["trust"] = trust
        st["confidence"] = confidence
        st["last_anom"] = x

        return {
            "ema_short": st["ema_s"],
            "ema_long": st["ema_l"],
            "delta": delta,
            "suspicion": suspicion,
            "trust": trust,
            "confidence": confidence,
            "rounds": st["rounds"],
        }

    def get_trust(self, cid): 
      return self.state[cid]["trust"]
    def get_confidence(self, cid): 
      return self.state[cid]["confidence"]
    def get_suspicion(self, cid): 
      return self.state[cid]["suspicion"]
    def snapshot(self): return self.state
