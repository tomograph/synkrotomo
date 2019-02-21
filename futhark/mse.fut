let main [n] (sane:[n]f32) (test:[n]f32) =
  let se = map2 (\s t -> (s-t)**2 ) sane test
  in (reduce (+) 0.0f32 se)/r32(n)
