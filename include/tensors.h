//
// Created by mohamedhany on 27/01/26.
//

#ifndef TENSORS_H
#define TENSORS_H
#include <algorithm>
#include <stdexcept>
#include <vector>
using namespace std;

unsigned int compute_sizes(vector<int>& dims, int i)
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
    if (s1.size() > s2.size())
    {
        big = s1;
        small = s2;
    } else
    {
        big = s2;
        small = s1;
    }

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
    vector<T>data;
    size_t sz = 1;
    vector <int> sizes;
    vector<bool> broadcasted_ranks;

    friend unsigned int compute_sizes(vector<int>& dims, int i);
public:
    Tensor(vector<int> shape)
    {
        this->shape = shape;
        sizes.push_back(1);
        for (unsigned int i = 0; i<this->shape.size(); i++)
        {
            sz*=this->shape[i];
            sizes.push_back(compute_sizes(this->shape, i+1));
        }
        data.resize(sz);
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
        return data[pointer];
    }

    T& at(int i)
    {
        if (i <0 || i > sz-1)
        {
            throw out_of_range("index for a tensor with shape, out of range");
        }
        return data[i % data.size()];
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
        data.clear();
        for (int i=0; i<v.size(); i++)
        {
            data.push_back(v[i]);
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
        data[pointer] = val;

    }

    void scale(const T val)
    {
        for (int i=0; i<data.size(); i++)
        {
            data[i]*=val;
        }
    }
    

    Tensor<T> add(Tensor<T>& other)
    {
        size_t big_sz;

        if (!is_shape_equal(this->shape, other.shape))
        {
            if (!can_broadcast(this->shape, other.shape) && !can_broadcast(other.shape, this->shape))
            {
                throw invalid_argument("can't add two unbroadcastable tensors");
            }

            if (other.shape.size() < this->shape.size())
            {
                big_sz = this->data.size();
                other.broadcast_to(*this);
            } else if (this->shape.size() < other.shape.size())
            {
                big_sz = other.data.size();
                this->broadcast_to(other);
            }
        } else
        {
            big_sz = this->data.size();
        }

        Tensor<T> res(this->shape);
        vector<int> indices;
        for (int i=0; i<big_sz; i++)
        {
            indices = res.flat_to_coupled_idx(i);
            res.at(indices) = other.at(indices) + this->at(indices);
        }
        return  res;
    }

    Tensor<T> multiply(const Tensor<T>& other)
    {
        size_t big_sz;

        if (!is_shape_equal(this->shape, other.shape))
        {
            if (!can_broadcast(this->shape, other.shape) && !can_broadcast(other.shape, this->shape))
            {
                throw invalid_argument("can't multiply two unbroadcastable tensors");
            }

            if (other.shape.size() < this->shape.size())
            {
                big_sz = this->data.size();
                other.broadcast_to(*this);
            } else if (this->shape.size() < other.shape.size())
            {
                big_sz = other.data.size();
                this->broadcast_to(other);
            }
        } else
        {
            big_sz = this->data.size();
        }

        Tensor<T> res(this->shape);
        vector<int> indices;
        for (int i=0; i<big_sz; i++)
        {
            indices = res.flat_to_coupled_idx(i);
            res.at(indices) = other.at(indices) * this->at(indices);
        }
        return  res;
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
            cout<<shape[i] << "x";
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
        for (int i=0; i< data.size()-1; i++)
        {
            cout << data[i] << ", ";
        }
        cout << data[data.size()-1] << "]";
        cout << endl;
    }


    //assumes the other tensor is the bigger one
    void broadcast_to(Tensor<T>& other)
    {
        if (other.shape.size() < this->shape.size())
        {
            throw invalid_argument("can't broadcast a bigger tensor into a smaller shape");
        }

        if (!can_broadcast(other.shape, this->shape))
        {
            throw invalid_argument("can't broadcast two tensors with incompaitable shapes");
        }
        vector<int> thisShapeCpy = this->shape;
        vector<int> otherShapeCpy = other.shape;

        broadcasted_ranks.resize(other.shape.size());
        sizes.resize(other.shape.size()+1);
        reverse(thisShapeCpy.begin(), thisShapeCpy.end());
        reverse(otherShapeCpy.begin(), otherShapeCpy.end());

        int diff = otherShapeCpy.size() - thisShapeCpy.size();
        for (int i=0; i<diff; i++)
        {
            sizes[thisShapeCpy.size()] = 0;
            broadcasted_ranks[otherShapeCpy.size() - thisShapeCpy.size() - 1] = true;
            thisShapeCpy.push_back(otherShapeCpy[thisShapeCpy.size()]);

        }

        reverse(thisShapeCpy.begin(), thisShapeCpy.end());
        for (int i=0; i<thisShapeCpy.size(); i++)
        {
            if (thisShapeCpy[i] == 1)
            {
                broadcasted_ranks[i] = true;
                sizes[thisShapeCpy.size()-i-1] = 0;
            }
        }
        this->shape = other.shape;
    }

    void reshape(const vector<int>& s)
    {
        int n_sz = 1;
        for (int i=0; i<s.size(); i++)
        {
            n_sz*=s[i];
        }
        if (n_sz != data.size())
        {
            throw invalid_argument("can't reshape tensor to incompaitable shape");
        }

        this->shape = s;
        sizes.clear();
        sizes.push_back(1);
        for (unsigned int i = 0; i<this->shape.size(); i++)
        {
            sizes.push_back(compute_sizes(this->shape, i+1));
        }

    }

};



#endif //TENSORS_H
