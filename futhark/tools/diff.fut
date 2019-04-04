let main [n] (sane:[n]f32) (test:[n]f32) =
  let std_dev = 0.05f32
  let bools = map2 (\s t -> if f32.abs(s-t) < std_dev then true else false) sane test
  in reduce (&&) true bools
  -- let out = filter (>0) <| map3 (\s t i -> if f32.abs(s-t) > std_dev then i else (-1)) sane test (iota n)
  -- -- in reduce (+) 0 diff
  -- in length out
