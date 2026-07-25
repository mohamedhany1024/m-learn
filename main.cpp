#include <iostream>
#include "include/tensors.h"
int main()
{
    Tensor<int> t1({2, 3, 4});
    Tensor<int> t2({1, 4});
    Tensor<int> t4({4});
    t1.shape_print();
    t1.sizes_print();
    vector<int> v1 = {1, 2, 4, 16, 3, 1, 6, 8, 17, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    t1.set_all(v1);
    vector<int> v2 = {1, 2, 3, 4};
    t2.set_all(v2);
    t1.data_print();
    cout << t1.at({0, 0, 0}) << endl;
    cout << t1.at({0, 0, 1}) << endl;
    cout << t1.at({0, 1, 1}) << endl;
    cout << t1.at({1, 1, 1}) << endl;
    t1.set({1, 1, 0}, -60);
    t1.data_print();
    vector<int> s1 = {5, 3, 2};
    vector<int> s2 = {2, 2};
    cout << can_broadcast(s1, s2);
    t1.shape_print();
    t1.sizes_print();
    //t2.broadcast_to(t1);
    t2.shape_print();
    t2.sizes_print();
    //cout << t2.at({1, 2, 0}) << endl;


    Tensor<int> t3({5, 1, 2});
    vector<int> v3 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    t3.set_all(v3);
    //t3.shape_print();
    //t3.sizes_print();
    //cout << t3.at({4, 0, 1}) << endl;
    t4.set_all(v2);
    t4 = t4.broadcast_to(t1);
    t4.shape_print();
    t4.sizes_print();
    cout << t4.at({1, 2, 0});

    auto soso = t1.add(t2);
    soso.data_print();

    auto hadamard_prod = t1.multiply(t2);
    cout << "hadamard_prod: " << endl;
    hadamard_prod.data_print();

    Tensor<int> t5({1, 2});
    Tensor<int> t6({2, 1});
    vector<int> data5 = {1, 2};
    vector<int> data6 = {2, 3};
    t5.set_all(data5);
    t6.set_all(data6);

    Tensor<int> t7 = t5.matmul(t6);
    t7.data_print();





    return 0;
}