#ifndef _SI_SC_IntegralEquation_HeaderFile
#define _SI_SC_IntegralEquation_HeaderFile

#include <GridFaceData.cuh>
#include <GridEdgeData.hxx>
#include <GridVertexData.hxx>

#include <BaseFunctionDefine.hxx>


//---------------------------------Elec   Edge-------------------------------------->>>
void Compute_DualContourValue_SI_SC(GridEdgeData* theData, 
				    Standard_Integer theFacePhysIndex,
				    Standard_Real& DualContourValue);

void Advance_OneElecElem_SI_SC(GridEdgeData* theData, 
			       Standard_Real Dt, 
			       Standard_Real si_scale,
			       Standard_Integer theEdgePhysIndex,
			       Standard_Integer theCurrentIndex,
			       Standard_Integer theFacePhysIndex);

void Advance_ElecElems_SI_SC(vector<GridEdgeData*>& theDatas,
			     Standard_Real Dt, 
			     Standard_Real si_scale,
			     Standard_Integer theEdgePhysIndex, 
			     Standard_Integer theCurrentIndex, 
			     Standard_Integer theFacePhysIndex);

void Advance_OneElecElem_SI_SC_Damping_1(GridEdgeData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex);

void Advance_OneElecElem_SI_SC_Damping_2(GridEdgeData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Real preEdgePhysData);

void Advance_OneElecElem_SI_SC_Damping_3(GridEdgeData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer thePREIndex);

void Advance_OneElecElem_SI_SC_Damping_4(GridEdgeData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Integer thePREIndex);

void Advance_OneElecElem_SI_SC_Damping(GridEdgeData* theData, 
				       Standard_Real Dt, 
				       Standard_Real si_scale,
				       Standard_Integer theEdgePhysIndex,
				       Standard_Integer theCurrentIndex,
				       Standard_Integer theFacePhysIndex,
				       Standard_Real damping_scale,
				       Standard_Integer theAEIndex,
				       Standard_Integer theBEIndex);

void Advance_ElecElems_SI_SC_Damping(vector<GridEdgeData*>& theDatas,
				     Standard_Real Dt, 
				     Standard_Real si_scale,
				     Standard_Integer theEdgePhysIndex,
				     Standard_Integer theCurrentIndex,
				     Standard_Integer theFacePhysIndex,
				     Standard_Real damping_scale,
				     Standard_Integer theAEIndex,
				     Standard_Integer theBEIndex);

void Advance_OneElecElem_SI_SC_Damping_new(GridEdgeData* theData, 
					   Standard_Real Dt, 
					   Standard_Real si_scale,
					   Standard_Integer theEdgePhysIndex,
					   Standard_Integer theCurrentIndex,
					   Standard_Integer theFacePhysIndex,
					   Standard_Real damping_scale,
					   Standard_Integer theAEIndex,
					   Standard_Integer theBEIndex,
					   Standard_Integer thePREIndex);

void Advance_ElecElems_SI_SC_Damping_new(vector<GridEdgeData*>& theDatas,
					 Standard_Real Dt, 
					 Standard_Real si_scale,
					 Standard_Integer theEdgePhysIndex,
					 Standard_Integer theCurrentIndex,
					 Standard_Integer theFacePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Integer thePREIndex);
//---------------------------------Elec   Edge--------------------------------------<<<


//---------------------------------Elec   Edge Along the Axis-------------------------------------->>>
void Compute_DualContourValue_SI_SC_AlongAxis(GridEdgeData* theDatas, 
				    Standard_Integer theFacePhysIndex,
				    Standard_Real& DualContourValue);

void Advance_OneElecElem_SI_SC_AlongAxis(vector<GridEdgeData>* theDatas, 
			       Standard_Size m_PhiNumber,
			       Standard_Real Dt, 
			       Standard_Real si_scale,
			       Standard_Integer theEdgePhysIndex,
			       Standard_Integer theCurrentIndex,
			       Standard_Integer theFacePhysIndex);

void Advance_ElecElems_SI_SC_AlongAxis(vector<GridEdgeData*>& theDatas,
			     Standard_Size m_PhiNumber,
			     Standard_Real Dt, 
			     Standard_Real si_scale,
			     Standard_Integer theEdgePhysIndex, 
			     Standard_Integer theCurrentIndex, 
			     Standard_Integer theFacePhysIndex);

void Advance_OneElecElem_SI_SC_AlongAxis_Damping_1(vector<GridEdgeData*> theData, 
				         Standard_Size m_PhiNumber,
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex);

void Advance_OneElecElem_SI_SC_AlongAxis_Damping_2(vector<GridEdgeData*> theDatas, 
				         Standard_Size m_PhiNumber,
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 vector<Standard_Real> preEdgePhysData);


void Advance_OneElecElem_SI_SC_AlongAxis_Damping(vector<GridEdgeData*> theData, 
				       Standard_Size m_PhiNumber,
				       Standard_Real Dt, 
				       Standard_Real si_scale,
				       Standard_Integer theEdgePhysIndex,
				       Standard_Integer theCurrentIndex,
				       Standard_Integer theFacePhysIndex,
				       Standard_Real damping_scale,
				       Standard_Integer theAEIndex,
				       Standard_Integer theBEIndex);

void Advance_ElecElems_SI_SC_AlongAxis_Damping(vector<GridEdgeData*>& theDatas,
				     Standard_Size m_PhiNumber,
				     Standard_Real Dt, 
				     Standard_Real si_scale,
				     Standard_Integer theEdgePhysIndex,
				     Standard_Integer theCurrentIndex,
				     Standard_Integer theFacePhysIndex,
				     Standard_Real damping_scale,
				     Standard_Integer theAEIndex,
				     Standard_Integer theBEIndex);
//---------------------------------Elec   Edge Along the Axis--------------------------------------<<<






//---------------------------------Elec Vertex-------------------------------------->>>
void Compute_DualContourValue_SI_SC(GridVertexData* theData, 
				    Standard_Integer theFacePhysIndex,
				    Standard_Real& DualContourValue);

void Advance_OneElecElem_SI_SC(GridVertexData* theData, 
			       Standard_Real Dt, 
			       Standard_Real si_scale,
			       Standard_Integer theEdgePhysIndex,
			       Standard_Integer theCurrentIndex,
			       Standard_Integer theFacePhysIndex);

void Advance_ElecElems_SI_SC(vector<GridVertexData*>& theDatas,
			     Standard_Real Dt, 
			     Standard_Real si_scale,
			     Standard_Integer theEdgePhysIndex, 
			     Standard_Integer theCurrentIndex, 
			     Standard_Integer theFacePhysIndex);

void Advance_OneElecElem_SI_SC_Damping_1(GridVertexData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex);

void Advance_OneElecElem_SI_SC_Damping_2(GridVertexData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Real preEdgePhysData);

void Advance_OneElecElem_SI_SC_Damping_3(GridVertexData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer thePREIndex);

void Advance_OneElecElem_SI_SC_Damping_4(GridVertexData* theData, 
					 Standard_Integer theEdgePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Integer thePREIndex);

void Advance_OneElecElem_SI_SC_Damping(GridVertexData* theData, 
				       Standard_Real Dt, 
				       Standard_Real si_scale,
				       Standard_Integer theEdgePhysIndex,
				       Standard_Integer theCurrentIndex,
				       Standard_Integer theFacePhysIndex,
				       Standard_Real damping_scale,
				       Standard_Integer theAEIndex,
				       Standard_Integer theBEIndex);

void Advance_ElecElems_SI_SC_Damping(vector<GridVertexData*>& theDatas,
				     Standard_Real Dt, 
				     Standard_Real si_scale,
				     Standard_Integer theEdgePhysIndex,
				     Standard_Integer theCurrentIndex,
				     Standard_Integer theFacePhysIndex,
				     Standard_Real damping_scale,
				     Standard_Integer theAEIndex,
				     Standard_Integer theBEIndex);

void Advance_OneElecElem_SI_SC_Damping_new(GridVertexData* theData, 
					   Standard_Real Dt, 
					   Standard_Real si_scale,
					   Standard_Integer theEdgePhysIndex,
					   Standard_Integer theCurrentIndex,
					   Standard_Integer theFacePhysIndex,
					   Standard_Real damping_scale,
					   Standard_Integer theAEIndex,
					   Standard_Integer theBEIndex,
					   Standard_Integer thePREIndex);

void Advance_ElecElems_SI_SC_Damping_new(vector<GridVertexData*>& theDatas,
					 Standard_Real Dt, 
					 Standard_Real si_scale,
					 Standard_Integer theEdgePhysIndex,
					 Standard_Integer theCurrentIndex,
					 Standard_Integer theFacePhysIndex,
					 Standard_Real damping_scale,
					 Standard_Integer theAEIndex,
					 Standard_Integer theBEIndex,
					 Standard_Integer thePREIndex);
//---------------------------------Elec Vertex--------------------------------------<<<











//---------------------------------Mag-------------------------------------->>>
void Compute_ContourValue_SI_SC(GridFaceData* theData, 
				Standard_Integer theEdgePhysIndex,
				Standard_Real& ContourValue);

void Advance_OneMagElem_SI_SC(GridFaceData* theData,
			      Standard_Real Dt, 
			      Standard_Real si_scale,
			      Standard_Integer theFacePhysIndex,
			      Standard_Integer theCurrentIndex, 
			      Standard_Integer theEdgePhysIndex);

void Advance_MagElems_SI_SC(vector<GridFaceData*>& theDatas, 
			    Standard_Real Dt, 
			    Standard_Real si_scale,
			    Standard_Integer theFacePhysIndex, 
			    Standard_Integer theCurrentIndex, 
			    Standard_Integer theEdgePhysIndex);

void Compute_ContourValue_SI_SC(GridEdgeData* theData, 
				Standard_Integer theEdgePhysIndex,
				Standard_Real& ContourValue);

void Advance_OneMagElem_SI_SC(GridEdgeData* theData,
			      Standard_Real Dt, 
			      Standard_Real si_scale,
			      Standard_Integer theFacePhysIndex,
			      Standard_Integer theCurrentIndex, 
			      Standard_Integer theEdgePhysIndex);

void Advance_MagElems_SI_SC(vector<GridEdgeData*>& theDatas, 
			    Standard_Real Dt, 
			    Standard_Real si_scale,
			    Standard_Integer theFacePhysIndex, 
			    Standard_Integer theCurrentIndex, 
			    Standard_Integer theEdgePhysIndex);
//---------------------------------Mag--------------------------------------<<<










#endif
