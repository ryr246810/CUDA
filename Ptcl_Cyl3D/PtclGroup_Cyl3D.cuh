#ifndef PTCLGROUP_Cyl3D
#define PTCLGROUP_Cyl3D

#include <vector>
#include <TxVector2D.h>
#include <TxVector.h>
#include <IndexAndWeights_Cyl3D.cuh>
#include <iostream>


// #ifdef __CUDA__

class PtclGroup_Cyl3D{
public:
	TxVector<double> * m_position;
	IndexAndWeights_Cyl3D    * m_idwt;
	TxVector<double>   * m_velocity;
	double *m_weight;
	int * m_state;
	int * m_rm_flag;
	int m_size;
	
	PtclGroup_Cyl3D(int n_ptcls){
		m_position  = new TxVector<double>[n_ptcls];
		m_idwt      = new IndexAndWeights_Cyl3D[n_ptcls];
		m_velocity = new TxVector<double>[n_ptcls];
		m_state = new int[n_ptcls];
		m_rm_flag = new int[n_ptcls];
		m_weight= new double[n_ptcls];
		m_size = 0;
	}
	~PtclGroup_Cyl3D(){
		delete[] m_position;      // 粒子位置？
		delete[] m_idwt;          // 什么意思？？？
		delete[] m_velocity;      // 粒子速度？
		delete[] m_weight;        // 粒子权重？
		delete[] m_state;         // 粒子状态？
		delete[] m_rm_flag;       // 粒子移除标志 
	}
	
	int get_size(){
		return m_size;
	}
	
	void add_ptcl(double x[3], const IndexAndWeights_Cyl3D& idwt, double v[3], int ptcl_state){
		m_position[m_size] = TxVector<double>(x);
		m_idwt[m_size] = idwt;
		m_velocity[m_size] = TxVector<double>(v);
		m_state[m_size] = ptcl_state;
		m_rm_flag[m_size] = 0;
		m_weight[m_size] = 1.0;
		m_size++;
	}
	void add_ptcl(double x[3], const IndexAndWeights_Cyl3D& idwt, double v[3], int ptcl_state,double weight){
		m_position[m_size] = TxVector<double>(x);
		m_idwt[m_size] = idwt;
		m_velocity[m_size] = TxVector<double>(v);
		m_state[m_size] = ptcl_state;
		m_rm_flag[m_size] = 0;
		m_weight[m_size] = weight;
		m_size++;
	}
	
	void remove_ptcl(int i){
		int last_index = m_size - 1;
		if(i != last_index){
			m_position[i] = m_position[last_index];
			m_idwt[i] = m_idwt[last_index];
			m_velocity[i] = m_velocity[last_index];
			m_weight[i] = m_weight[last_index];
			m_state[i] = m_state[last_index];
			m_rm_flag[i] = m_rm_flag[last_index];
		}
		m_size--;
	}

	void resort_ptcl(){
		if(m_size > 2){
			// swap(m_position[0], m_position[m_size-1]);
			// swap(m_idwt[0],     m_idwt[m_size-1]);
			// swap(m_velocity[0], m_velocity[m_size-1]);
			// swap(m_weight[0],   m_weight[m_size-1]);
			// swap(m_state[0],    m_state[m_size-1]);
			// swap(m_rm_flag[0],  m_rm_flag[m_size-1]);

			TxVector<double> m_positionTmp = m_position[0];
			IndexAndWeights_Cyl3D m_idwtTmp = m_idwt[0];
			TxVector<double> m_velocityTmp = m_velocity[0];
			double m_weightTmp = m_weight[0];
			int m_stateTmp = m_state[0];
			int m_rm_flagTmp = m_rm_flag[0];

			m_position[0] = m_position[m_size-1];
			m_idwt[0] = m_idwt[m_size-1];
			m_velocity[0] = m_velocity[m_size-1];
			m_weight[0] = m_weight[m_size-1];
			m_state[0] = m_state[m_size-1];
			m_rm_flag[0] = m_rm_flag[m_size-1];

			m_position[m_size-1] = m_positionTmp;
			m_idwt[m_size-1] = m_idwtTmp;
			m_velocity[m_size-1] = m_velocityTmp;
			m_weight[m_size-1] = m_weightTmp;
			m_state[m_size-1] = m_stateTmp;
			m_rm_flag[m_size-1] = m_rm_flagTmp;

		}
		
		// for(int i = 0; i < m_size / 2; ++i){
		// 	swap(m_position[i], m_position[m_size - i - 1]);
		// 	swap(m_idwt[i], m_idwt[m_size - i - 1]);
		// 	swap(m_velocity[i], m_velocity[m_size - i - 1]);
		// 	swap(m_weight[i], m_weight[m_size - i - 1]);
		// 	swap(m_state[i], m_state[m_size - i - 1]);
		// 	swap(m_rm_flag[i], m_rm_flag[m_size - i - 1]);
		// }
	}

	void reInit_velocity(){
		for(int i = 0; i < m_size; ++i){
			m_velocity[i] = TxVector<double>(0.0, 0.0, 0.0);
		}
	}
	
	//引用类型其实就是别名，编译器会自动处理
	TxVector<double>& position(int i){ 
		return m_position[i]; 
	}
	
	IndexAndWeights_Cyl3D& idwt(int i){
		return m_idwt[i];
	}
	
	TxVector<double>& velocity(int i){ 
		return m_velocity[i]; 
	}

	TxVector<double>* Get_velocity(){
		return m_velocity;
	}

	double& weight(int i){
		return m_weight[i];
	}
	int& state(int i){
		return m_state[i];
	}
	int& b_rm_flag(int i){
		return m_rm_flag[i];
	}
	int* Get_rm_flag(){
		return m_rm_flag;
	}

};

// #endif
#endif
