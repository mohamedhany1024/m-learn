//
// Created by mohamedhany on 27/01/26.
//

#ifndef TENSORS_H
#define TENSORS_H
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace std;

unsigned int compute_sizes(const vector<int>& dims, int i)
{
    int res = dims[dims.size()-1];
    if (i == 1)
    {
        return res;
    }
    for (int j=1; j<i; j++)
    {
        res*=dims[dims.size()-1-j];
    }
    return res;

}

bool can_broadcast(vector<int> s1, vector<int> s2)
{
    vector<int> big, small;
    bool flag = true;
    /*if (s1.size() > s2.size())
    {
        big = s1;
        small = s2;
    } else
    {
        big = s2;
        small = s1;
    }*/
    small = s1;
    big = s2;

    reverse(big.begin(), big.end());
    reverse(small.begin(), small.end());
    int diff = big.size() - small.size();
    for (int i=0; i<diff; i++)
    {
        small.push_back(big[small.size()]);
    }

    for (int i=0; i<big.size(); i++)
    {
        if (small[i] != big[i] && small[i] != 1)
        {
            flag = false;
        }
    }

    return flag;
}

bool is_shape_equal(vector<int> s1, vector<int> s2)
{
    bool flag = true;

    if (s1.size()!= s2.size())
    {
        return false;
    }

    for (int i=0; i<s1.size(); i++)
    {
        if (s1[i] != s2[i])
        {
            flag = false;

        }
    }
    return flag;
}


template <class T>
class Tensor
{
    vector<int> shape;
    //vector<T>data;
    shared_ptr<vector<T>> data;
    size_t sz = 1;
    vector <int> sizes;
    vector<bool> broadcasted_ranks;

    friend unsigned int compute_sizes(vector<int>& dims, int i);
public:
    Tensor(const vector<int>& shape)
    {
        this->shape = shape;
        sizes.push_back(1);
        for (unsigned int i = 0; i<this->shape.size(); i++)
        {
            sz*=this->shape[i];
            sizes.push_back(compute_sizes(this->shape, i+1));
        }
        data = std::make_shared<vector<T>>(sz);
        //data->resize(sz);
        broadcasted_ranks.resize(shape.size());
    }

    Tensor(shared_ptr<vector<T>> prevData,const vector<int> & shape)
    {
        this->shape = shape;
        sizes.push_back(1);
        for (unsigned int i = 0; i<this->shape.size(); i++)
        {
            sz*=this->shape[i];
            sizes.push_back(compute_sizes(this->shape, i+1));
        }

        if (prevData == nullptr)
        {
            throw invalid_argument("Tensor initialization failed: either provided data is null or data size is not compaitable with shape.");
        }
        data = prevData;
        broadcasted_ranks.resize(shape.size());

    }

    T& at(const vector<int>& indices)
    {
        int pointer = 0;
        //Access Exceptions
        if (indices.size() != shape.size())
        {
            throw invalid_argument("e: tried to access tensor of a specific shape with an incompaitable shape");
        }

        for (int i=0; i<indices.size(); i++)
        {
            if (indices[i] > shape[i]-1 || indices[i] < 0)
            {
                throw out_of_range("index for a tensor with shape, out of range");
            }
            pointer += indices[i]*sizes[sizes.size()-2-i];
        }
        //cout << "pointer is: " << pointer << endl;
        return (*data)[pointer];
    }

    T& at(int i)
    {
        if (i <0 || i > sz-1)
        {
            throw out_of_range("index for a tensor with shape, out of range");
        }
        return (*data)[i % data->size()];
    }

    vector<int> flat_to_coupled_idx(int idx)
    {
        vector<int> indices(this->shape.size(), 0);
        for (int i=0; i<indices.size(); i++)
        {
            indices[i] = idx/sizes[sizes.size()-2-i];
            idx = idx % sizes[sizes.size()-2-i];
        }
        return indices;
    }

    void set_all(vector<T>& v)
    {
        if (v.size() != sz) {
            throw invalid_argument("Vector size does not match tensor capacity.");
        }
        data->clear();
        for (int i=0; i<v.size(); i++)
        {
            data->push_back(v[i]);
        }
    }

    void set(const vector<int>& indices, const T val)
    {
        if (indices.size() != shape.size())
        {
            throw invalid_argument("e: tried to access tensor of shape with an incompaitable index");
        }
        int pointer = 0;
        for (int i=0; i<indices.size(); i++)
        {
            if (indices[i] > shape[i]-1 || indices[i] < 0)
            {
                throw out_of_range("index for a tensor with shape shape_print() , out of range");
            }
            pointer += indices[i]*sizes[sizes.size()-2-i];
        }
        (*data)[pointer] = val;

    }

    void zeros()
    {
        data->clear();
        for (int i=0; i<sz; i++)
        {
            data->push_back(0);
        }
    }

    void scale(const T val)
    {
        for (int i=0; i<data->size(); i++)
        {
            (*data)[i]*=val;
        }
    }
    

    Tensor<T> add(Tensor<T>& other)
    {
        size_t big_sz;
        Tensor<T> thisTensorView (data, this->shape);
        Tensor<T> otherTensorView (other.data, other.shape);

        if (!is_shape_equal(this->shape, other.shape))
        {
            if (can_broadcast(this->shape, other.shape))
            {
                // 'this' can adapt to 'other's shape
                big_sz = other.data->size();
                thisTensorView = this->broadcast_to(other);
            }
            else if (can_broadcast(other.shape, this->shape))
            {
                // 'other' can adapt to 'this' shape
                big_sz = this->data->size();
                otherTensorView = other.broadcast_to(*this);
            }
            else
            {
                throw invalid_argument("can't add two unbroadcastable tensors");
            }
        } else {
            big_sz = this->data->size();
        }

        Tensor<T> res(thisTensorView.shape);
        vector<int> indices;
        for (int i=0; i<big_sz; i++)
        {
            indices = res.flat_to_coupled_idx(i);
            res.at(indices) = otherTensorView.at(indices) + thisTensorView.at(indices);
        }
        return res;
    }

    Tensor<T> multiply( Tensor<T>& other)
    {
        size_t big_sz;
        Tensor<T> thisTensorView (data, this->shape);
        Tensor<T> otherTensorView (other.data, other.shape);
        if (!is_shape_equal(this->shape, other.shape))
        {
            if (can_broadcast(this->shape, other.shape))
            {
                // 'this' can adapt to 'other's shape
                big_sz = other.data->size();
                thisTensorView = this->broadcast_to(other);
            }
            else if (can_broadcast(other.shape, this->shape))
            {
                // 'other' can adapt to 'this' shape
                big_sz = this->data->size();
                otherTensorView = other.broadcast_to(*this);
            }
            else
            {
                throw invalid_argument("can't multiply two unbroadcastable tensors");
            }
        } else {
            big_sz = this->data->size();
        }

        Tensor<T> res(thisTensorView.shape);
        vector<int> indices;
        for (int i=0; i<big_sz; i++)
        {
            indices = res.flat_to_coupled_idx(i);
            res.at(indices) = otherTensorView.at(indices) * thisTensorView.at(indices);
        }
        return  res;
    }

    Tensor<T> matmul(Tensor<T>& other)
    {
        if (this->shape.size() != 2 || other.shape.size() != 2)
        {
            throw invalid_argument("can't apply matmul on non-rank-2 tensor.");
        }

        if (this->shape[1] != other.shape[0])
        {
            throw invalid_argument("invalid matmul: incompaitable shapes");
        }

        Tensor<T> res({this->shape[0], other.shape[1]});
        res.zeros();

        // a direct code translation for: $\text{for each element in the product mat C resulted from A*B, Where A is an n x m mat and B is an m x l matrix, where m is the common index size} C_{i j} = \sum_{k=0}^{m}(a_{i k} * b_{k j})$
        for (int i=0; i<res.shape[0]; i++)
        {
            for (int j=0; j<res.shape[1]; j++)
            {
                //shared index
                for (int k=0; k<this->shape[1]; k++)
                {
                    res.at({i, j}) += (this->at({i, k}) * other.at({k, j}));
                }
            }
        }
        return res;
    }

    vector<int> get_shape() const
    {
        return this->shape;
    }

    void shape_print()
    {
        cout << "shape:";
        for (int i=0; i<shape.size(); i++)
        {
            cout<<shape[i] << ((i==shape.size()-1) ? "" : "x");
        }
        cout << endl;
    }

    void sizes_print()
    {
        cout << "sizes of dimensions:" << endl;
        for (int i=0; i<sizes.size(); i++)
        {
            cout << "size of " << i << "D: " << sizes[i] << endl;
        }
    }

    void data_print()
    {
        cout << "[";
        for (int i=0; i< data->size()-1; i++)
        {
            cout << (*data)[i] << ", ";
        }
        cout << (*data)[data->size()-1] << "]";
        cout << endl;
    }


    //assumes the other tensor is the bigger one
    Tensor<T> broadcast_to(const Tensor<T>& other)
    {
        if (other.shape.size() < this->shape.size())
        {
            throw invalid_argument("can't broadcast a bigger tensor into a smaller shape");
        }

        if (!can_broadcast(other.shape, this->shape) && !can_broadcast(this->shape, other.shape))
        {
            throw invalid_argument("can't broadcast two tensors with incompaitable shapes");
        }
        vector<int> thisShapeCpy = this->shape;
        vector<int> otherShapeCpy = other.shape;
        Tensor<T> res(data, this->shape);

        res.broadcasted_ranks.resize(other.shape.size());
        res.sizes.resize(other.shape.size()+1);
        reverse(thisShapeCpy.begin(), thisShapeCpy.end());
        reverse(otherShapeCpy.begin(), otherShapeCpy.end());

        // sets the sizes/strides for the ranks that are missing from the smaller tensor to zero
        int diff = otherShapeCpy.size() - thisShapeCpy.size();
        for (int i=0; i<diff; i++)
        {
            res.sizes[thisShapeCpy.size()] = 0;
            res.broadcasted_ranks[otherShapeCpy.size() - thisShapeCpy.size() - 1] = true;
            //thisShapeCpy.push_back(otherShapeCpy[thisShapeCpy.size()]);
            thisShapeCpy.push_back(1);

        }

        reverse(thisShapeCpy.begin(), thisShapeCpy.end());
        for (int i=0; i<thisShapeCpy.size(); i++)
        {
            if (thisShapeCpy[i] == 1)
            {
                res.broadcasted_ranks[i] = true;
                res.sizes[thisShapeCpy.size()-i-1] = 0;
            }
        }
        res.shape = other.shape;
        res.sz = 1;
        for (int i=0; i<res.shape.size(); i++)
        {
            res.sz*=res.shape[i];
        }
        return res;
    }

    Tensor<T> reshape(const vector<int>& s)
    {
        Tensor<T> res(shape, data);
        int n_sz = 1;
        for (int i=0; i<s.size(); i++)
        {
            n_sz*=s[i];
        }
        if (n_sz != data->size())
        {
            throw invalid_argument("can't reshape tensor to incompaitable shape");
        }

        res.shape = s;
        res.sizes.clear();
        res.sizes.push_back(1);
        for (unsigned int i = 0; i<this->shape.size(); i++)
        {
            res.sizes.push_back(compute_sizes(this->shape, i+1));
        }
        return res;
    }

};

#endif //TENSORS_H