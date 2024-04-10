class OpenOrders: 
    def __init__(self, contract):
        self.contract_name = contract
        self.num_open_orders = 0
        self.price_to_id = {} # price to id dict
        self.id_to_price = {} # id to price dict
        self.id_to_qty = {} # id to qty dict
    
    ## adjust for each 
    def adjust_qty(self, id, adj):
        self.id_to_qty[id] += adj

        # deleting order
        if self.id_to_qty[id] == 0:
            self.num_open_orders -= 1
            price = self.id_to_price[id]

            try:
                del self.id_to_price[id]
                del self.price_to_id[price]
                del self.id_to_qty[id]
            except Exception as e:
                print("Error (0) deleting filled order: ",e)
                
    def add_order(self, price, id, qty):
        if not price in self.price_to_id:
            self.price_to_id[price] = id
            self.num_open_orders += 1
        if not id in self.id_to_qty:
            self.id_to_qty[id] = qty
        if not id in self.id_to_price:
            self.id_to_price[id] = price

    def modify_order(self,price,qty,old_id,new_id):
        # create order (if there is no order with matching ID)
        if (old_id == new_id):
            if not old_id in self.id_to_price:
                self.id_to_price[old_id] = price
                self.price_to_id[price] = old_id
                self.id_to_qty[old_id] = qty
                self.num_open_orders += 1
            # update order
            else:
                # delete old price to data
                try:
                    del self.price_to_id[self.id_to_price[old_id]]
                except Exception as e:
                    print("Error (1) deleting filled order: ",e)

                # add new price to id
                self.price_to_id[price] = old_id
                self.id_to_price[old_id] = price
                self.id_to_qty[old_id] = qty
        else:
            if not old_id in self.id_to_price:
                self.id_to_price[new_id] = price
                self.price_to_id[price] = new_id
                self.id_to_qty[new_id] = qty
                self.num_open_orders += 1
            else:
                # old order still exists so delete and then update with new values

                # delete old price, id, and qty
                try:
                    del self.price_to_id[self.id_to_price[old_id]] # error is no price in price_to_id for old price
                    del self.id_to_price[old_id]
                    del self.id_to_qty[old_id]
                except Exception as e:
                    print("Error (2) deleting filled order: ",e)

                # add new price to new id
                self.price_to_id[price] = new_id
                self.id_to_price[new_id] = price
                self.id_to_qty[new_id] = qty


    # getting the quantity based on the price
    def get_qty(self, price):
        p_id = self.price_to_id[price]
        return self.id_to_qty[p_id]

    def get_id(self, price):
        return self.price_to_id[price]