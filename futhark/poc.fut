let main [n] (inarr1:[n]f32) (iters:i32)  =
  let res = loop (inarr1) = (inarr1) for i < iters do
    scatter (replicate n 0.0f32) (rotate (7) (iota n)) inarr1
    -- map (\i -> i+1) inarr1


    -- let test1 = scatter (replicate n 0.0f32) (rotate (7) (iota n)) test
    -- let test2 = scatter (replicate n 0.0f32) (rotate (7) (iota n)) test1
    -- in map (\i -> i+1)  test
    -- map (\i -> f32.sqrt(i*7)) inarr1
    -- in scan (+) 0.0f32 <| map (\i -> f32.sqrt(i*7)) test
  in res
  -- let test4 = scatter (replicate (length test3) 0.0f32) (rotate (7) (iota n)) test3
  -- in reduce ( map2 (+) ) (replicate n 0.0f32) test1
  -- let res2 = map (\c -> reduce (+) 0.0f32 <| map (\r -> unsafe res[r,c]) (iota iters) ) (iota n)
  -- let res2 =
  -- let r = reduce (+) 0.0 (flatten res)
  -- in replicate n r
  -- in scan (+) 0.0f32 (flatten res)
