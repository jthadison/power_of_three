"""OANDA broker integration for live execution"""
import oandapyV20
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
from typing import Dict, Optional

class OandaBroker:
    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.client = oandapyV20.API(access_token=api_key, environment=environment)
    
    def place_order(self, signal: Dict) -> Optional[str]:
        """Place order based on signal"""
        try:
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": self._convert_symbol(signal['symbol']),
                    "units": str(int(signal['position_size'])),
                    "stopLossOnFill": {
                        "price": str(signal['stop_loss'])
                    },
                    "takeProfitOnFill": {
                        "price": str(signal['take_profit_1'])
                    }
                }
            }
            
            if signal['signal_type'] == 'short':
                order_data['order']['units'] = f"-{order_data['order']['units']}"
            
            r = orders.OrderCreate(self.account_id, data=order_data)
            response = list(self.client.request(r))
            
            return response[0]['orderCreateTransaction']['id'] if response else None
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def _convert_symbol(self, symbol: str) -> str:
        """Convert your symbols to OANDA format"""
        symbol_map = {
            'US30': 'US30_USD',
            'NAS100': 'NAS100_USD', 
            'SPX500': 'SPX500_USD',
            'XAUUSD': 'XAU_USD'
        }
        return symbol_map.get(symbol, symbol)
    
    def get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            r = accounts.AccountDetails(self.account_id)
            response = list(self.client.request(r))
            return float(response[0]['account']['balance']) if response else 0.0
        except:
            return 0.0