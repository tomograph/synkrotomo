let main [n] (inarr1:[n]f32) (inarr2:[n]f32) =
  let arr1 = map2 (\s t -> s * t) inarr1 inarr2
  in map2 (\a1 a2 -> a1+a2) arr1 inarr1
