#ifndef math_Array2D
#define math_Array2D

#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define HOST_DEVICE2 __host__ __device__

template<class T> class Array2D
{
public:
	HOST_DEVICE2 Array2D() {m_HaveInit = false; m_Array = NULL; m_Row = m_Column = 0;}
	HOST_DEVICE2 void InitArray(int row, int column){
		if(m_Array != NULL)
			delete[] m_Array;
		m_Array = new T[row * column];
		m_HaveInit = true;
		m_Row = row;
		m_Column = column;
	}
	
	HOST_DEVICE2 int GetIndex(int row, int column) const{
		assert(m_HaveInit);
		assert(row < m_Row);
		assert(column < m_Column);
		return column * m_Row + row;
	}
	
	HOST_DEVICE2 void SetValue(int row, int column, T value) const{
		m_Array[GetIndex(row, column)] = value;
	}

	HOST_DEVICE2 void SetAllValue(T value) const{
		for(int i = 0; i < m_Row; ++i)
			for(int j = 0; j < m_Column; ++j)
				m_Array[GetIndex(i, j)] = value;
	}

	HOST_DEVICE2 void AddValue(int row, int column, T value) const{
		T tmp = m_Array[GetIndex(row, column)];
		m_Array[GetIndex(row, column)] = value + tmp;
	}
	
	HOST_DEVICE2 T GetValue(int row, int column) const{
		int index = GetIndex(row, column);
		return m_Array[index];
	}
	
	HOST_DEVICE2 T *GetElemAddr(int row, int column) const{
		int index = GetIndex(row, column);
		return (m_Array + index);
	}

	HOST_DEVICE2 T* GetArray() const {
		return m_Array;
	}

	HOST_DEVICE2 int GetSize() const {
		return m_Row * m_Column;
	}
	
	HOST_DEVICE2 void ClearArray(){
		if(!m_HaveInit)
			return;
		m_HaveInit = false;
		m_Row = m_Column = 0;
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

};


#endif