let main [n] (inarr1:[n]f32) (iters:i32) : [n]f32 =
  let res = loop (inarr1) = (inarr1) for i < iters do
    scatter (replicate n 0.0f32) (rotate (7) (iota n)) inarr1
  in res
