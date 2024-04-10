from typing import Optional

from xchangelib import xchange_client
import asyncio
import orderbook
                           
                
## orders to    

CONTRACTS = ["EPT","DLO", "MKU", "IGM", "BRV", "SCP", "JAK", "JMS"]             

ORDER_SIZE = 65 # 50

ORDER_L1 = 15 # 25
ORDER_L2 = 10 # 10
ORDER_L3 = 5 # 5

L1_SPREAD = 0.02
L2_SPREAD = L1_SPREAD*2
L3_SPREAD = L1_SPREAD*3
L4_SPREAD = L1_SPREAD*4


class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)

    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("order fill", self.positions)

    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        # print("something was traded")
        pass

    async def bot_handle_book_update(self, symbol: str) -> None:
        # print("book update")
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        await asyncio.sleep(5)
        print("attempting to trade")

        ### 
        # 5 stocks - EPT, DLO, MKU, IGM, BRV
        # 2 etf - SCP, JAK 
        # 1 risk free asset - JMS
        # 
        # 
        # 
        # 
        # ###
    


        # Viewing Positions
        print("My positions:", self.positions)

    async def view_books(self):
        """Prints the books every 3 seconds."""
        while True:
            await asyncio.sleep(3)
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def start(self):
        """
        Creates tasks that can be run in the background. Then connects to the exchange
        and listens for messages.
        """
        

        # trading day (0-251)
        self.day = 0
        self.rain = []
        self.fairs = {}
        self.order_book = {}
        self.pos = {}
        self.order_ids = {}

        self.open_orders = {}


        for month in CONTRACTS:
            # TODO make other (for different levels of orders)
            self.order_ids[month+' bid'] = ''
            self.order_ids[month+' ask'] = ''

            self.order_ids[month+' l1 bid'] = ''
            self.order_ids[month+' l1 ask'] = ''

            self.order_ids[month+' l2 bid'] = ''
            self.order_ids[month+' l2 ask'] = ''

            self.order_ids[month+' l3 bid'] = ''
            self.order_ids[month+' l3 ask'] = ''

            self.order_ids[month+' l4 bid'] = ''
            self.order_ids[month+' l4 ask'] = ''


            self.fairs[month] = start_fair

            self.order_book[month] = {
                'Best Bid':{'Price':0,'Quantity':0},
                'Best Ask':{'Price':0,'Quantity':0}
                }

            self.pos[month] = 0

            self.open_orders[month] = OpenOrders(month)
                    
        asyncio.create_task(self.trade())
        # asyncio.create_task(self.view_books())
        await self.connect()


async def main():
    SERVER = 'staging.uchicagotradingcompetition.com:3333' # run on sandbox
    my_client = MyXchangeClient(SERVER,"princeton","nidoqueen-chansey-9131")
    await my_client.start()
    return

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
    
    
    
    
    
    ##
# 'spoofing'
# penny ordering and profitting off of small margins
#  update quotes and keep data
# keep track of order book, our orders, and open orders

##