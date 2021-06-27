def multiply(filename_a, shape_a, dtype ='float64', offset_a=0, order_a='F', transpose_a=False,
            filename_b=None, shape_b, offset_b=None, order_b='C', transpose_b=False,
            filename_out, offset_out=0, order_out='C', mode_out='r+'
            ):

    filename_b = filename_b or filename_a
    offset_b = offset_b or offset_a
    shape_b = shape_b or shape_a
    assert order_a in {'F','C'}, "order_a must be F or C"
    assert order_b in {'F','C'}, "order_b must be F or C"
    if order_a == 'C':
        order_a = 'F'
        shape_a = reverse(shape_a)
        transpose_a = not transpose_a
    if order_b == 'C':
        order_b = 'F'
        shape_b = reverse(shape_b)
        transpose_b = not transpose_b
    assert order_a == 'F' and order_b = 'F' # real assert

    if transpose_a:
        (row_count_out, extra_out_a) = shape_a[1], shape_a[0]
    else:
        (row_count_out, extra_out_a) = shape_a

    if transpose_b:
        (col_count_out, extra_out_b) = shape_b
    else:
        (col_count_out, extra_out_b) = shape_b[1], shape_b[0]

    assert extra_out_a == extra_out_b, "Expect a and b to have a common dimension that is multiplied along"





