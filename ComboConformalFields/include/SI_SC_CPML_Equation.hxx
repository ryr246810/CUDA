#ifndef _SI_SC_CPML_Equation_HeaderFile
#define _SI_SC_CPML_Equation_HeaderFile


#include <DataBase.hxx>
#include <BaseFunctionDefine.hxx>

#include <GridVertexData.hxx>
#include <GridEdgeData.hxx>
#include <GridFaceData.cuh>

//--------------------------------a and b-------------------------------------->>>
void Compute_a_b_SI_SC(DataBase* theData, 
		       const Standard_Integer theTruncDir,
		       const Standard_Real Dt, 
		       Standard_Real& a,
		       Standard_Real& b);


void Compute_Dual_a_b_SI_SC(DataBase* theData, 
			    const Standard_Integer theTruncDir, 
			    const Standard_Real Dt, 
			    Standard_Real& a, 
			    Standard_Real& b);

void Get_a_b_SI_SC(DataBase* theData, 
		   const Standard_Integer theTruncDir,
		   const Standard_Real Dt, 
		   Standard_Real& a,
		   Standard_Real& b);


void Compute_DualContourValue1_SI_SC(GridEdgeData* theData,
				     Standard_Integer theFacePhysIndex,
				     Standard_Real& DualContourValue);

void Compute_PE1_SI_SC(GridEdgeData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePE1Index,
		       Standard_Real DualContourValue);

void Advance_CPML_OneElecElem_SI_SC(GridEdgeData* theData,
				    Standard_Real Dt, 
				    Standard_Real si_scale,
				    Standard_Integer theEdgePhysIndex,
				    Standard_Integer thePE1Index,
				    Standard_Integer thePE2Index,
				    Standard_Integer theFacePhysIndex);

void Advance_CPML_ElecElems_SI_SC(vector<GridEdgeData*>& theDatas,
				  Standard_Real Dt, 
				  Standard_Real si_scale,
				  Standard_Integer theEdgePhysIndex,
				  Standard_Integer thePE1Index,
				  Standard_Integer thePE2Index,
				  Standard_Integer theFacePhysIndex);

void Advance_CPML_OneElecElem_SI_SC_Damping(GridEdgeData* theData,
					    Standard_Real Dt, 
					    Standard_Real si_scale,
					    Standard_Integer theEdgePhysIndex,
					    Standard_Integer thePE1Index,
					    Standard_Integer thePE2Index,
					    Standard_Integer theFacePhysIndex,
					    Standard_Real damping_scale,
					    Standard_Integer theAEIndex,
					    Standard_Integer theBEIndex);

void Advance_CPML_ElecElems_SI_SC_Damping(vector<GridEdgeData*>& theDatas,
					  Standard_Real Dt, 
					  Standard_Real si_scale,
					  Standard_Integer theEdgePhysIndex,
					  Standard_Integer thePE1Index,
					  Standard_Integer thePE2Index,
					  Standard_Integer theFacePhysIndex,
					  Standard_Real damping_scale,
					  Standard_Integer theAEIndex,
					  Standard_Integer theBEIndex);


void Advance_CPML_OneElecElem_SI_SC_Damping_new(GridEdgeData* theData,
						Standard_Real Dt, 
						Standard_Real si_scale,
						Standard_Integer theEdgePhysIndex,
						Standard_Integer thePE1Index,
						Standard_Integer thePE2Index,
						Standard_Integer theFacePhysIndex,
						Standard_Real damping_scale,
						Standard_Integer theAEIndex,
						Standard_Integer theBEIndex,
						Standard_Integer thePREIndex);

void Advance_CPML_ElecElems_SI_SC_Damping_new(vector<GridEdgeData*>& theDatas,
					      Standard_Real Dt, 
					      Standard_Real si_scale,
					      Standard_Integer theEdgePhysIndex,
					      Standard_Integer thePE1Index,
					      Standard_Integer thePE2Index,
					      Standard_Integer theFacePhysIndex,
					      Standard_Real damping_scale,
					      Standard_Integer theAEIndex,
					      Standard_Integer theBEIndex,
					      Standard_Integer thePREIndex);

void Compute_DualContourValue1_SI_SC(GridVertexData* theData,
				     Standard_Integer theMagPhysIndex,
				     Standard_Real& DualContourValue);


void Compute_DualContourValue2_SI_SC(GridVertexData* theData, 
				     Standard_Integer theMagPhysIndex,
				     Standard_Real& DualContourValue);

void Compute_PE1_SI_SC(GridVertexData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePE1Index,
		       Standard_Real DualContourValue);

void Compute_PE2_SI_SC(GridVertexData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePE2Index,
		       Standard_Real DualContourValue);

void Advance_CPML_OneElecElem_SI_SC(GridVertexData* theData,
				    Standard_Real Dt, 
				    Standard_Real si_scale,
				    Standard_Integer theElecPhysIndex,
				    Standard_Integer thePE1Index,
				    Standard_Integer thePE2Index,
				    Standard_Integer theMagPhysIndex);

void Advance_CPML_ElecElems_SI_SC(vector<GridVertexData*>& theDatas,
				  Standard_Real Dt, 
				  Standard_Real si_scale,
				  Standard_Integer theElecPhysIndex,
				  Standard_Integer thePE1Index,
				  Standard_Integer thePE2Index,
				  Standard_Integer theMagPhysIndex);


void Advance_CPML_OneElecElem_SI_SC_Damping(GridVertexData* theData,
					    Standard_Real Dt, 
					    Standard_Real si_scale,
					    Standard_Integer theElecPhysIndex,
					    Standard_Integer thePE1Index,
					    Standard_Integer thePE2Index,
					    Standard_Integer theMagPhysIndex,
					    Standard_Real damping_scale,
					    Standard_Integer theAEIndex,
					    Standard_Integer theBEIndex);


void Advance_CPML_ElecElems_SI_SC_Damping(vector<GridVertexData*>& theDatas,
					  Standard_Real Dt, 
					  Standard_Real si_scale,
					  Standard_Integer theElecPhysIndex,
					  Standard_Integer thePE1Index,
					  Standard_Integer thePE2Index,
					  Standard_Integer theMagPhysIndex,
					  Standard_Real damping_scale,
					  Standard_Integer theAEIndex,
					  Standard_Integer theBEIndex);


void Advance_CPML_OneElecElem_SI_SC_Damping_new(GridVertexData* theData,
						Standard_Real Dt, 
						Standard_Real si_scale,
						Standard_Integer theElecPhysIndex,
						Standard_Integer thePE1Index,
						Standard_Integer thePE2Index,
						Standard_Integer theMagPhysIndex,
						Standard_Real damping_scale,
						Standard_Integer theAEIndex,
						Standard_Integer theBEIndex,
						Standard_Integer thePREIndex);


void Advance_CPML_ElecElems_SI_SC_Damping_new(vector<GridVertexData*>& theDatas,
					      Standard_Real Dt, 
					      Standard_Real si_scale,
					      Standard_Integer theElecPhysIndex,
					      Standard_Integer thePE1Index,
					      Standard_Integer thePE2Index,
					      Standard_Integer theMagPhysIndex,
					      Standard_Real damping_scale,
					      Standard_Integer theAEIndex,
					      Standard_Integer theBEIndex,
					      Standard_Integer thePREIndex);


void Compute_ContourValue1_SI_SC(GridFaceData* theData, 
				 Standard_Integer theElecPhysIndex,
				 Standard_Real& contourValue);

void Compute_ContourValue2_SI_SC(GridFaceData* theData,
				 Standard_Integer theElecPhysIndex,
				 Standard_Real& contourValue);

void Compute_PM1_SI_SC(GridFaceData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePM1Index, 
		       Standard_Real contourValue);

void Compute_PM2_SI_SC(GridFaceData* theData, 
		       Standard_Real Dt, 
		       Standard_Integer thePM2Index,
		       Standard_Real contourValue);

void Advance_CPML_OneMagElem_SI_SC(GridFaceData* theData, 
				   Standard_Real Dt, 
				   Standard_Real si_scale,
				   Standard_Integer theMagPhysIndex,
				   Standard_Integer thePM1Index, 
				   Standard_Integer thePM2Index, 
				   Standard_Integer theElecPhysIndex);

void Advance_CPML_MagElems_SI_SC(vector<GridFaceData*>& theDatas, 
				 Standard_Real Dt, 
				 Standard_Real si_scale,
				 Standard_Integer theMagPhysIndex, 
				 Standard_Integer thePM1Index, 
				 Standard_Integer thePM2Index, 
				 Standard_Integer theElecPhysIndex);


void Compute_ContourValue1_SI_SC(GridEdgeData* theData, 
				 Standard_Integer theElecPhysIndex,
				 Standard_Real& contourValue);

void Compute_PM1_SI_SC(GridEdgeData* theData,
		       Standard_Real Dt,
		       Standard_Integer thePM1Index, 
		       Standard_Real contourValue);

void Advance_CPML_OneMagElem_SI_SC(GridEdgeData* theData, 
				   Standard_Real Dt, 
				   Standard_Real si_scale,
				   Standard_Integer theMagPhysIndex,
				   Standard_Integer thePM1Index, 
				   Standard_Integer thePM2Index, 
				   Standard_Integer theElecPhysIndex);

void Advance_CPML_MagElems_SI_SC(vector<GridEdgeData*>& theDatas, 
				 Standard_Real Dt, 
				 Standard_Real si_scale,
				 Standard_Integer theMagPhysIndex, 
				 Standard_Integer thePM1Index, 
				 Standard_Integer thePM2Index, 
				 Standard_Integer theElecPhysIndex);


///////////////////////////////////  Axis ////////////////////////////



void Compute_DualContourValue1_SI_SC_AlongAxis(GridEdgeData* theData,
                                     Standard_Integer theMagPhysIndex,
                                     Standard_Real& DualContourValue);


void Compute_PE1_SI_SC_AlongAxis(vector<GridEdgeData*> theDatas,
                       Standard_Size m_PhiNumber,
                       Standard_Real Dt,
                       Standard_Integer thePE1Index,
                       Standard_Real DualContourValue);


void Advance_CPML_OneElecElem_SI_SC_AlongAxis(vector<GridEdgeData*> theDatas,
                                    Standard_Size m_PhiNumber,
                                    Standard_Real Dt,
                                    Standard_Real si_scale,
                                    Standard_Integer theElecPhysIndex,
                                    Standard_Integer thePE1Index,
                                    Standard_Integer thePE2Index,
                                    Standard_Integer theMagPhysIndex);


void Advance_CPML_ElecElems_SI_SC_AlongAxis(vector<GridEdgeData*>& theDatas,
                                  Standard_Size m_PhiNumber,
                                  Standard_Real Dt,
                                  Standard_Real si_scale,
                                  Standard_Integer theElecPhysIndex,
                                  Standard_Integer thePE1Index,
                                  Standard_Integer thePE2Index,
                                  Standard_Integer theMagPhysIndex);


void Advance_CPML_OneElecElem_SI_SC_AlongAxis_Damping(vector<GridEdgeData*> theDatas,
                                            Standard_Size m_PhiNumber,
                                            Standard_Real Dt,
                                            Standard_Real si_scale,
                                            Standard_Integer theElecPhysIndex,
                                            Standard_Integer thePE1Index,
                                            Standard_Integer thePE2Index,
                                            Standard_Integer theMagPhysIndex,
                                            Standard_Real damping_scale,
                                            Standard_Integer theAEIndex,
                                            Standard_Integer theBEIndex);


void Advance_CPML_ElecElems_SI_SC_AlongAxis_Damping(vector<GridEdgeData*>& theDatas,
                                          Standard_Size m_PhiNumber,
                                          Standard_Real Dt,
                                          Standard_Real si_scale,
                                          Standard_Integer theElecPhysIndex,
                                          Standard_Integer thePE1Index,
                                          Standard_Integer thePE2Index,
                                          Standard_Integer theMagPhysIndex,
                                          Standard_Real damping_scale,
                                          Standard_Integer theAEIndex,
                                          Standard_Integer theBEIndex);


#endif
