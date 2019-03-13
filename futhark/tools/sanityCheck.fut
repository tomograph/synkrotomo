let main [n] (sane:[n]f32) (test:[n]f32) =
  let std_dev = 0.05f32
  let bools = map2 (\s t -> if f32.abs(s-t) < std_dev then true else false) sane test
  in reduce (&&) true bools
  -- let diff = map2 (\s t -> if f32.abs(s-t) < std_dev then 1i32 else 0i32) sane test
  -- in reduce (+) 0
