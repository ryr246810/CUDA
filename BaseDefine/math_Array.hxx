#ifndef math_Array
#define math_Array

#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <CUDAHeader.cuh>

#define HOST_DEVICE1 __host__ __device__

template<class T> class Array3D
{
public:
	HOST_DEVICE1 Array3D() {m_HaveInit = false; m_Array = NULL; m_Row = m_Column = m_Height = 0;}
	HOST_DEVICE1 void InitArray(int row, int column, int height){
		if(m_Array != NULL)
			delete[] m_Array;
		m_Array = new T[row * column * height];
		m_HaveInit = true;
		m_Row = row;
		m_Column = column;
		m_Height = height;
	}
	
	HOST_DEVICE1 int GetIndex(int row, int column, int height) const{
		assert(m_HaveInit);
		assert(row < m_Row);
		assert(column < m_Column);
		assert(height < m_Height);
		return height * m_Column * m_Row + column * m_Row + row;
	}
	
	HOST_DEVICE1 void SetValue(int row, int column, int height, T value) const{
		m_Array[GetIndex(row, column, height)] = value;
	}

	HOST_DEVICE1 void AddValue(int row, int column, int height, T value) const{
		T tmp = m_Array[GetIndex(row, column, height)];
		m_Array[GetIndex(row, column, height)] = value + tmp;
	}
	
	HOST_DEVICE1 T GetValue(int row, int column, int height) const{
		int index = GetIndex(row, column, height);
		return m_Array[index];
	}
	
	HOST_DEVICE1 T *GetElemAddr(int row, int column, int height) const{
		int index = GetIndex(row, column, height);
		return (m_Array + index);
	}

	HOST_DEVICE1 T* GetArray() const {
		return m_Array;
	}

	HOST_DEVICE1 int GetSize() const {
		return m_Row * m_Column * m_Height;
	}
	
	HOST_DEVICE1 void ClearArray(){
		if(!m_HaveInit)
			return;
		m_HaveInit = false;
		m_Row = m_Column = m_Height = 0;
		if(m_Array != NULL)
		{
			delete[] m_Array;
			m_Array = NULL;
		}
	}

public:
	bool m_HaveInit;// 判断该类是否已经初始化
	T *m_Array;// 存放数据的数组 
	int m_Row;// 行数
	int m_Column;// 列数
	int m_Height;// 高数
};


#endif