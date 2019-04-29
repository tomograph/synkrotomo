let main [n] (inarr1:[n]f32) (inarr2:[n]f32) (iters:i32)  =
  let test = map (\_ -> let arr1 = map2 (\s t -> s * t) inarr1 inarr2
    let arr2 =  map2 (\a1 a2 -> a1+a2) (reverse inarr1) inarr2
    let arr3 =  map2 (\a1 a2 -> a1+a2) (reverse inarr2) inarr1
    let arr4 =  map2 (\a1 a2 ->  a1-a2) arr2 arr3
    in map2 (\a1 a2 -> a2-a1) arr1 arr4
  ) (iota iters)
  let test1 = map (\r -> scatter (replicate n 0.0f32) (rotate (n/2) (iota n)) r ) test
  in reduce ( map2 (+) ) (replicate n 0.0f32) test1
  -- let res2 = map (\c -> reduce (+) 0.0f32 <| map (\r -> unsafe res[r,c]) (iota iters) ) (iota n)
  -- let res2 =
  -- let r = reduce (+) 0.0 (flatten res)
  -- in replicate n r
  -- in scan (+) 0.0f32 (flatten res)
