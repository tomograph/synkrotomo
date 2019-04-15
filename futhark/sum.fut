let main [n] (inarr1:[n]f32) (inarr2:[n]f32) =
  let arr1 = map2 (\s t -> s * t) inarr1 inarr2
  let arr2 = map2 (\a1 a2 -> a1+a2) (reverse inarr1) inarr2
  let arr3 = map2 (\a1 a2 -> a1+a2) (reverse inarr2) inarr1
  let arr4 = map2 (\a1 a2 -> a1*a2) arr2 arr3
  in map2 (\a1 a2 -> a1+a2) arr1 arr4
