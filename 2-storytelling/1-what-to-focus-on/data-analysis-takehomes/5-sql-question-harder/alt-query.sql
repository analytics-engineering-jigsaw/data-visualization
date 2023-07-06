with orders_daily as (
  select
    customer_id,
    date(order_placed_at) as date_,
    sum(revenue) as revenue
  from orders
  group by 1, 2
),
order_window as (
  select
    date(date_spine.date_) as date_,
    orders_daily.customer_id,
    orders_daily.revenue,
    sum(revenue) over(partition by customer_id order by date_spine.date_ rows between 28 preceding and current row) as prev_28,
    sum(revenue) over(partition by customer_id order by date_spine.date_ rows between current row and 28 following) as next_28
  from date_spine
  left join orders_daily on date(date_spine.date_) = date(orders_daily.date_)
)
select * from order_window where customer_id is not null