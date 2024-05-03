#include <algorithm>
#include "Species_Cyl3D.cuh"
#include "IndexAndWeights_Cyl3D.cuh"
#include "NodeField_Cyl3D.cuh"
#include "CUDAHeader.cuh"

float Species_Cyl3D::Advance_With_Cuda(double dt)
{

	// float time = advance_cuda(dt);
	float time = advance_cuda_nonstream(dt);
	// printf("Test....... %f\n", ptcl_cuda[0][1].m_weight);

	return time;
}

void Species_Cyl3D::CopyPtclDatas_HostToDevice(int idx)
{
	size_t TxVectorSizeBytes = sizeof(TxVector<double>) * ptcl_grps[idx]->get_size();

	cudaMemcpy(dev_ptclIndxWeights[idx], &(ptcl_grps[idx]->m_idwt[0]), ptcl_grps[idx]->get_size() * sizeof(IndexAndWeights_Cyl3D), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_position[idx], &(ptcl_grps[idx]->m_position[0]), TxVectorSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_velocity[idx], &(ptcl_grps[idx]->m_velocity[0]), TxVectorSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weight[idx], &(ptcl_grps[idx]->m_weight[0]), ptcl_grps[idx]->get_size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rm_flag[idx], &(ptcl_grps[idx]->m_rm_flag[0]), ptcl_grps[idx]->get_size() * sizeof(int), cudaMemcpyHostToDevice);
}

void Species_Cyl3D::CopyPtclDatas_HostToDevice(cudaStream_t *streams, int idx)
{
	size_t TxVectorSizeBytes = sizeof(TxVector<double>) * ptcl_grps[idx]->get_size();

	cudaMemcpyAsync(dev_ptclIndxWeights[idx], &(ptcl_grps[idx]->m_idwt[0]), ptcl_grps[idx]->get_size() * sizeof(IndexAndWeights_Cyl3D), cudaMemcpyHostToDevice, streams[idx]);
	cudaMemcpyAsync(dev_position[idx], &(ptcl_grps[idx]->m_position[0]), TxVectorSizeBytes, cudaMemcpyHostToDevice, streams[idx]);
	cudaMemcpyAsync(dev_velocity[idx], &(ptcl_grps[idx]->m_velocity[0]), TxVectorSizeBytes, cudaMemcpyHostToDevice, streams[idx]);
	cudaMemcpyAsync(dev_weight[idx], &(ptcl_grps[idx]->m_weight[0]), ptcl_grps[idx]->get_size() * sizeof(double), cudaMemcpyHostToDevice, streams[idx]);
	cudaMemcpyAsync(dev_rm_flag[idx], &(ptcl_grps[idx]->m_rm_flag[0]), ptcl_grps[idx]->get_size() * sizeof(int), cudaMemcpyHostToDevice, streams[idx]);
}

void Species_Cyl3D::CopyPtclDatas_DeviceToHost(int idx)
{
	size_t TxVectorSizeBytes = sizeof(TxVector<double>) * ptcl_grps[idx]->get_size();

	cudaMemcpy(&(ptcl_grps[idx]->m_idwt[0]), dev_ptclIndxWeights[idx], ptcl_grps[idx]->get_size() * sizeof(IndexAndWeights_Cyl3D), cudaMemcpyDeviceToHost);
	cudaMemcpy(&(ptcl_grps[idx]->m_velocity[0]), dev_velocity[idx], TxVectorSizeBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(&(ptcl_grps[idx]->m_position[0]), dev_position[idx], TxVectorSizeBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(&(ptcl_grps[idx]->m_rm_flag[0]), dev_rm_flag[idx], ptcl_grps[idx]->get_size() * sizeof(int), cudaMemcpyDeviceToHost);
}

void Species_Cyl3D::CopyPtclDatas_DeviceToHost(cudaStream_t *streams, int idx)
{
	size_t TxVectorSizeBytes = sizeof(TxVector<double>) * ptcl_grps[idx]->get_size();

	cudaMemcpyAsync(&(ptcl_grps[idx]->m_idwt[0]), dev_ptclIndxWeights[idx], ptcl_grps[idx]->get_size() * sizeof(IndexAndWeights_Cyl3D), cudaMemcpyDeviceToHost, streams[idx]);
	cudaMemcpyAsync(&(ptcl_grps[idx]->m_velocity[0]), dev_velocity[idx], TxVectorSizeBytes, cudaMemcpyDeviceToHost, streams[idx]);
	cudaMemcpyAsync(&(ptcl_grps[idx]->m_position[0]), dev_position[idx], TxVectorSizeBytes, cudaMemcpyDeviceToHost, streams[idx]);
	cudaMemcpyAsync(&(ptcl_grps[idx]->m_rm_flag[0]), dev_rm_flag[idx], ptcl_grps[idx]->get_size() * sizeof(int), cudaMemcpyDeviceToHost, streams[idx]);
}

void Species_Cyl3D::fill_with_PtclDatas(int grpsNum, int maxPtclNum)
{
	size_t TxVectorSizeBytes = sizeof(TxVector<double>) * maxPtclNum;

	for (int i = 0; i < grpsNum; ++i)
	{
		cudaMalloc((void **)&(dev_ptclIndxWeights[i]), sizeof(IndexAndWeights_Cyl3D) * maxPtclNum);
		cudaMalloc((void **)&(dev_efileds[i]), TxVectorSizeBytes);
		cudaMalloc((void **)&(dev_bfileds[i]), TxVectorSizeBytes);
		cudaMalloc((void **)&(dev_position[i]), TxVectorSizeBytes);
		cudaMalloc((void **)&(dev_velocity[i]), TxVectorSizeBytes);
		cudaMalloc((void **)&(dev_weight[i]), sizeof(double) * maxPtclNum);
		cudaMalloc((void **)&(dev_rm_flag[i]), sizeof(int) * maxPtclNum);
	}
}

void Species_Cyl3D::free_PtclDatas(int grpsNum)
{

	for (int i = 0; i < grpsNum; ++i)
	{
		cudaFree(dev_ptclIndxWeights[i]);
		cudaFree(dev_efileds[i]);
		cudaFree(dev_bfileds[i]);
		cudaFree(dev_position[i]);
		cudaFree(dev_velocity[i]);
		cudaFree(dev_weight[i]);
		cudaFree(dev_rm_flag[i]);
	}
}

void Species_Cyl3D::Resize_PtclDatas(int n_grp)
{

	dev_ptclIndxWeights.resize(n_grp);
	dev_efileds.resize(n_grp);
	dev_bfileds.resize(n_grp);
	dev_position.resize(n_grp);
	dev_velocity.resize(n_grp);
	dev_weight.resize(n_grp);
	dev_rm_flag.resize(n_grp);
}

void Species_Cyl3D::Translate_RemovePtcl(PtclGroup_Cyl3D *ptcl_grp)
{
	int np = ptcl_grp->get_size();
	int flag;
	int *rm_flag = ptcl_grp->Get_rm_flag();

	for (int i = 0; i < np; i += flag)
	{

		flag = 1;
		if (!rm_flag[i])
			continue;
		else
		{
			ptcl_grp->remove_ptcl(i);
			if (i != np - 1)
				rm_flag[i] = rm_flag[np - 1];
			np--;
			flag = 0;
		}
	}
}