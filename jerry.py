from typing import Optional

from xchangelib import xchange_client
import asyncio
import copy 

class MyXchangeClient(xchange_client.XChangeClient):
    '''A shell client with the methods that can be implemented to interact with the xchange.'''

    def compute_etf_fv(self):
        if self.fair_values['EPT'] == None: 
            return
        if self.fair_values['SCP'] != None and self.fair_values['BRV'] != None:
            self.fair_values['JCR'] = 0.3 * self.fair_values['EPT'] + 0.3 * self.fair_values['IGM'] + 0.4 * self.fair_values['BRV']
        if self.fair_values['DLO'] != None and self.fair_values['MKU'] != None:
            self.fair_values['JAK'] = 0.2 * self.fair_values['EPT'] + 0.5 * self.fair_values['DLO'] + 0.3 * self.fair_values['MKU']

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        
        self.symbols = xchange_client.SYMBOLS
        self.fade = {}
        
        self.fair_values = {}
        self.state = {}
        for symbol in self.symbols:
            self.fair_values[symbol] = 5000
            self.state[symbol] = {}
            self.fade[symbol] = 0

        self.etfs = ['SCP', 'JAK']
        self.etf_weights = [
            (0, 0, 0.3, 0, 0, 0.3, 0.4, 0),
            (0, 0, 0.2, 0.5, 0.3, 0, 0, 0)
        ]

        self.book_other = self.order_books
        # desired orders to have
        self.desired_orders = {} 

    def update_state(self): 
        # await asyncio.sleep(0.01)
        self.book_other = copy.deepcopy(self.order_books)
        print(self.open_orders)
        for order in self.open_orders.values():
            order_request, qty, is_market = order
            if is_market:
                continue
            symbol = order_request.symbol
            side = order_request.side
            # print(order_request.limit)
            px = order_request.limit.px
            qty = order_request.limit.qty

            if side == xchange_client.Side.BUY:
                self.book_other[symbol].bids[px] -= qty
            elif side == xchange_client.Side.SELL:
                self.book_other[symbol].asks[px] -= qty

        for security, book in self.book_other.items():
            sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
            sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
            if len(sorted_bids) > 0 and len(sorted_asks) > 0:
                self.fair_values[security] = (sorted_bids[-1][0] + sorted_asks[0][0]) / 2
                self.state[security] = {
                    'best_bid' : sorted_bids[-1], # (price, qty)
                    'best_ask' : sorted_asks[0], # (price, qty)
                    'mid_prc' : (sorted_bids[-1][0] + sorted_asks[0][0]) / 2
                }
        self.compute_etf_fv()
        return 
    
    def compute_fade(self):
        for security, pos in self.positions.items():
            if security == 'cash':
                continue
            self.fade[security] = pos

    async def market_making(self):
        await self.clear_pos()
        while True: 
            await asyncio.sleep(0.1)
            self.update_state()
            self.compute_fade()
            for symbol in self.symbols:
                if self.state[symbol] == {}:
                    continue
                if symbol in ['SCP', 'JMS', 'JAK']:
                    continue
                bb = self.state[symbol]['best_bid'][0]
                ba = self.state[symbol]['best_ask'][0]
                if ba - bb > 2:
                    print(bb, ba)
                    await self.place_order(symbol, 1, xchange_client.Side.BUY, bb+1-self.fade[symbol])
                    await self.place_order(symbol, 1, xchange_client.Side.SELL, ba-1-self.fade[symbol])
               
            

    async def arbitrage(self):
        await self.clear_pos()
        while True:
            await asyncio.sleep(0.05)
            
            flag = 0
            for security, pos in self.positions.items():
                if security == 'cash':
                    continue
                if abs(pos) >= 100:
                    flag = 1
            if flag:
                await self.clear_pos()


            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                if len(sorted_bids) > 0 and len(sorted_asks) > 0:
                    self.fair_values[security] = (sorted_bids[-1][0] + sorted_asks[0][0]) / 2
                    self.state[security] = {
                        'best_bid' : sorted_bids[-1], # (price, qty)
                        'best_ask' : sorted_asks[0], # (price, qty)
                        'mid_prc' : (sorted_bids[-1][0] + sorted_asks[0][0]) / 2
                    }
            self.compute_etf_fv()

            
            # check for creation opportunities
            flag = 0
            for symbol in self.state:
                if symbol == 'JMS':
                    continue
                if self.state[symbol] == {}:
                    flag = 1

            if flag == 1:
                continue

            
            create_arb_SCP = 10 * self.state['SCP']['best_bid'][0] - 3 * self.state['EPT']['best_ask'][0] - 3 * self.state['IGM']['best_ask'][0] - 4 * self.state['BRV']['best_ask'][0] 
            vol = min(self.state['EPT']['best_ask'][1]//3, self.state['IGM']['best_ask'][1]//3, self.state['BRV']['best_ask'][1]//4, self.state['SCP']['best_bid'][1]//10)
            vol = min(vol, 10)
            # vol = 1

            if create_arb_SCP > 0:
                await self.place_swap_order('toSCP', vol)
                await self.place_order('EPT', 3 * vol, xchange_client.Side.BUY)
                await self.place_order('IGM', 3 * vol, xchange_client.Side.BUY)
                await self.place_order('BRV', 4 * vol, xchange_client.Side.BUY)
                await self.place_order('SCP', 10 * vol, xchange_client.Side.SELL)

            redeem_arb_SCP = 3 * self.state['EPT']['best_bid'][0] + 3 * self.state['IGM']['best_bid'][0] + 4 * self.state['BRV']['best_bid'][0] - 10 * self.state['SCP']['best_ask'][0]
            vol = min(self.state['EPT']['best_bid'][1]//3, self.state['IGM']['best_bid'][1]//3, self.state['BRV']['best_bid'][1]//4, self.state['SCP']['best_ask'][1]//10)
            vol = min(vol, 10)
            # vol = 1
            if redeem_arb_SCP > 0:
                await self.place_swap_order('fromSCP', vol)
                await self.place_order('EPT', 3 * vol, xchange_client.Side.SELL)
                await self.place_order('IGM', 3 * vol, xchange_client.Side.SELL)
                await self.place_order('BRV', 4 * vol, xchange_client.Side.SELL)
                await self.place_order('SCP', 10 * vol, xchange_client.Side.BUY)
            

            # vol = 1
            create_arb_JAK = 10 * self.state['JAK']['best_bid'][0] - 2 * self.state['EPT']['best_ask'][0] - 5 * self.state['DLO']['best_ask'][0] - 3 * self.state['MKU']['best_ask'][0]
            vol = min(self.state['JAK']['best_bid'][1]//10, self.state['EPT']['best_ask'][1]//2, self.state['DLO']['best_ask'][1]//5, self.state['MKU']['best_ask'][1]//3)
            vol = min(vol, 10)
            # print(create_arb_JAK)
            if create_arb_JAK > 0:
                await self.place_swap_order('toJAK', vol)
                await self.place_order('JAK', 10 * vol, xchange_client.Side.SELL)
                await self.place_order('EPT', 2 * vol, xchange_client.Side.BUY)
                await self.place_order('DLO', 5 * vol, xchange_client.Side.BUY)
                await self.place_order('MKU', 3 * vol, xchange_client.Side.BUY)
            

            redeem_arb_JAK = -10 * self.state['JAK']['best_ask'][0] + 2 * self.state['EPT']['best_bid'][0] + 5 * self.state['DLO']['best_bid'][0] + 3 * self.state['MKU']['best_bid'][0]
            vol = min(self.state['JAK']['best_ask'][1]//10, self.state['EPT']['best_bid'][1]//2, self.state['DLO']['best_bid'][1]//5, self.state['MKU']['best_bid'][1]//3)
            vol = min(vol, 10)
            # print(redeem_arb_JAK)
            if redeem_arb_JAK > 0:
                await self.place_swap_order('fromJAK', vol)
                await self.place_order('JAK', 10 * vol, xchange_client.Side.BUY)
                await self.place_order('EPT', 2 * vol, xchange_client.Side.SELL)
                await self.place_order('DLO', 5 * vol, xchange_client.Side.SELL)
                await self.place_order('MKU', 3 * vol, xchange_client.Side.SELL)

            print(create_arb_SCP, redeem_arb_SCP, create_arb_JAK, redeem_arb_JAK)
            
            



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
        # await self.generate_orders()
        pass

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        # print("Swap response")
        pass


    async def trade(self):
        """This is a task that is started right before the bot connects and runs in the background."""
        await asyncio.sleep(5)
        print("attempting to trade")
        await self.place_order("BRV", 3, xchange_client.Side.SELL, 7)

        # Cancelling an order
        order_to_cancel = await self.place_order("BRV",3, xchange_client.Side.BUY, 5)
        await asyncio.sleep(5)
        await self.cancel_order(order_to_cancel)

        # Placing Swap requests
        await self.place_swap_order('toJAK', 1)
        await asyncio.sleep(5)
        await self.place_swap_order('fromSCP', 1)
        await asyncio.sleep(5)

        # Placing an order that gets rejected for exceeding qty limits
        await self.place_order("BRV",1000, xchange_client.Side.SELL, 7)
        await asyncio.sleep(5)

        # Placing a market order
        market_order_id = await self.place_order("BRV",10, xchange_client.Side.SELL)
        print("Market Order ID:", market_order_id)
        await asyncio.sleep(5)

        # Viewing Positions
        print("My positions:", self.positions)

    async def view_books(self):
        """Prints the books every 1 seconds."""
        while True:
            await asyncio.sleep(1)
            # self.log.append(self.order_books)
            # pickle.dump(self.log, open("log.pickle", "wb"))
            
            for security, book in self.order_books.items():
                sorted_bids = sorted((k,v) for k,v in book.bids.items() if v != 0)
                sorted_asks = sorted((k,v) for k,v in book.asks.items() if v != 0)
                print(f"Bids for {security}:\n{sorted_bids}")
                print(f"Asks for {security}:\n{sorted_asks}")

    async def clear_pos(self):
        
        for order in self.open_orders:
            print("Cancelling order", order)
            await self.cancel_order(order)

        # await asyncio.sleep(1)
        print("Pos", self.positions)
        for security, pos in self.positions.items():
            if security == 'cash':
                continue
            print(security, pos)
            if pos > 0:
                for i in range(pos):
                    await self.place_order(security, 1, xchange_client.Side.SELL)
            elif pos < 0:
                for i in range(-pos):
                    await self.place_order(security, 1, xchange_client.Side.BUY)
            
            print("Cleared positions", self.positions)
                
    async def start(self):
        """
        Creates tasks that can be run in the background. Then connects to the exchange
        and listens for messages.
        """
        
        # asyncio.create_task(self.arbitrage())
        asyncio.create_task(self.market_making())
        await self.connect()


async def main():
    SERVER = 'staging.uchicagotradingcompetition.com:3333' # run on sandbox
    my_client = MyXchangeClient(SERVER, "princeton","nidoqueen-chansey-9131")
    await my_client.start()
    return

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(main())
    