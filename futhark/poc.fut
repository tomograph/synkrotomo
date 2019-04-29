let main [n] (inarr1:[n]f32) (iters:i32)  =
  let test = scatter (replicate n 0.0f32) (rotate (7) (iota n)) inarr1
  let test1 = scatter (replicate n 0.0f32) (rotate (7) (iota n)) test
  let test2 = scatter (replicate n 0.0f32) (rotate (7) (iota n)) test1
  let test3 = map (\i -> f32.sqrt(i*7)) test2
  let test4 = scatter (replicate n 0.0f32) (rotate (7) (iota n)) test3
  in scan (+) 0.0f32 test4
  -- in reduce ( map2 (+) ) (replicate n 0.0f32) test1
  -- let res2 = map (\c -> reduce (+) 0.0f32 <| map (\r -> unsafe res[r,c]) (iota iters) ) (iota n)
  -- let res2 =
  -- let r = reduce (+) 0.0 (flatten res)
  -- in replicate n r
  -- in scan (+) 0.0f32 (flatten res)
