let main [n] (inarr1:[n]f32) (inarr2:[n]f32) =
  let res = map2 (\s t -> s + t) inarr1 inarr2
  in reduce (+) 0.0f32 res
