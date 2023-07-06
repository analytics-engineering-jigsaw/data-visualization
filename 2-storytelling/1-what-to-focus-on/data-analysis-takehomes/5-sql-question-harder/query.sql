SELECT
    o.order_id
  , o.customer_id
  , o.order_placed_at
  , o.revenue
  , SUM(prev_orders.revenue) AS prev_28_days_revenue
  , SUM(next_orders.revenue) AS next_28_days_revenue
FROM orders AS o
CROSS JOIN orders AS prev_orders
CROSS JOIN orders AS next_orders
WHERE (prev_orders.customer_id = o.customer_id
        AND prev_orders.order_placed_at BETWEEN date(o.order_placed_at,'-28 day') AND o.order_placed_at)
    AND (next_orders.customer_id = o.customer_id
        AND next_orders.order_placed_at BETWEEN o.order_placed_at AND date(o.order_placed_at,'+28 day'))
GROUP BY o.order_id
       , o.customer_id
       , o.order_placed_at
       , o.revenue


