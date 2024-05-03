#ifndef PTCLGEOM_Cyl3DBNDSINKER
#include <TxVector.h>
#include <TxVector2D.h>

#include <intersect_func_Cyl3D.cuh>
#include <GridGeometry_Cyl3D.hxx>
#include <ZRGrid.hxx>
// #include <blitz/array.h>
#include <math_Array.hxx>
#include <AppendingEdgeData.hxx>

// using namespace blitz;

class PtclBndInfo_Cyl3D{
public:	
	GridGeometry_Cyl3D * geometry;
	const ZRGrid * global_grid;
	Array3D<int> cell_type;
	// Array<int, 3> cell_type;// 0 : innner cell, 1: bnd cell
	Array3D<vector<EdgeData*> > bnd_edge;
	// Array<vector<EdgeData*>, 3> bnd_edge;	
	int n_cell_z, n_cell_r;
        int m_phi_number;	
	PtclBndInfo_Cyl3D(GridGeometry_Cyl3D * geom);
	
	void setup();
	
	bool check_sink(const TxVector<double>& start_pos,  const TxVector<double>& end_pos, int i_cell, int j_cell,int k_cell);
	
	bool check_sink2(const TxVector<double>& start_pos,  const TxVector<double>& end_pos, int i_cell, int j_cell,int k_cell);
	int check_sink3(const TxVector<double>& start_pos,  const TxVector<double>& end_pos, int i_cell, int j_cell,int k_cell);

	Array3D<int> Get_cell_type(){
		return cell_type;
	};
	
};

#define PTCLGEOMBNDSINKER
#endif
