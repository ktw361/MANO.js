import * as tf from '@tensorflow/tfjs';
import { Rodrigues_forward } from './mano.js';

function Rodrigues_forward_test() {
    const vec = tf.tensor([2.90096335, 5.67236714, 6.29441052]);
    const exp = tf.tensor([
        [-0.6936258,   0.07068303,  0.71685923],
        [0.70566722, - 0.13313241,  0.69592351],
        [0.14462718,  0.98857457 ,  0.04246537],
    ])
    const actual = Rodrigues_forward(vec);
    const res = tf.test_util.expectArraysClose(exp, actual);
    exp.print();
    actual.print();
}

function main() {
    Rodrigues_forward_test();
}

main();