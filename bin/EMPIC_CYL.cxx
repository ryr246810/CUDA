/* System Header */
#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string>

/* basedefine LIB Header */
#include <BaseDataDefine.hxx>
#include <PhysConsts.hxx>

/* ComboFields LIB Header */
#include <ComboFields_DefineCntr.hxx>
#include <ComboFieldsDefineRules.hxx>
#include <SI_SC_ComboEMFields.hxx>
#include <SI_SC_ComboEMFields_Cyl3D.hxx>

/* NodeFields LIB Header */
#include <NodeFlds_Elec.hxx>
#include <NodeFlds_Hdf5Output.hxx>
#include <NodeFldsTecio.hxx>
#include <NodeFldsTecio_Cyl3D.hxx>

/* GeomTxtBuilder LIB Header */
#include <Geom_TxtBuilders.hxx>

/* FieldsBase LIB Header */
#include <FieldsDefineCntr.hxx>
#include <FieldsDgnSets.hxx>

/* GridData LIB Header */
#include <ZRGrid.hxx>
#include <UnitsSystemDef.hxx>
#include <GridGeometry_Cyl3D.hxx>

/* GridGeneration LIB Header */
#include <ZRGrid_Ctrl.hxx>
#include <Model_Ctrl.hxx>
#include <Grid_Generation.hxx>
#include <Mesh_Write.hxx>
#include <GridGeom_Write.hxx>
#include <ReadStd.hxx>
#include <Grid_Tool.hxx>
#include <PadeGrid_Tool.hxx>
#include <UniformGrid_Tool.hxx>

/* MaterialDefineInterface LIB Header */
#include <MaterialDefineInterface.hxx>

/* OCAFTool LIB Header */
#include <OCAF_Object.hxx>
#include <OCAF_ObjectTool.hxx>
#include <OCAF_Application.hxx>
#include <OCAF_ColorMap.hxx>

/* Ptcl_Cyl3D LIB Header */
#include <Species_Cyl3D.cuh>
#include <NodeField_Cyl3D.cuh>
#include <PtclSource_Cyl3D.cuh>

/* ReadTool LIB Header */
#include <ReadTxFile.hxx>
#include <RunCmdLineArgs.hxx>

/* txbase LIB Header */
#include <TxStreams.h>
#include <TxHierAttribSet.h>
#include <TxMaker.h>
#include <TxMakerMap.h>

/* OCCT LIB Header */
#include <TDocStd_Document.hxx>
#include <TDF_ChildIterator.hxx>

/* tecIO LIB Header */
#include <TECIO.h>

/* Cuda_Files LIB Header */
#include "func.h"

// extern Standard_Real EMFields_Elapsed_cuda;

void ComputeCFLConstrain(const Standard_Real dl, Standard_Real &dt);

Standard_Boolean BeginReadOCCDocument();

Standard_Boolean ReadOCCDocument(OCAFDocumentCtrl *&theDocumentCtrl, Model_Ctrl *&theModelCtrl);

Standard_Boolean EndReadOCCDocument();

RunCmdLineArgs cmdLineArgs;
void CurrentTime(void)
{
	time_t now = time(0);
	char *date = ctime(&now);
	std::cout << "Current Time : " << date << std::endl;
}

enum FileType
{
	FULL = 0,
	GRID = 1,
	SOLUTION = 2
};

int main(int argc, char **argv)
{
	// 0. Test Kernel function of GPU
	std::cout << "\n0. Test Kernel function of GPU" << std::endl;
	// 此处已删除

	cmdLineArgs.setFromCmdLine(argc, argv);
	SetupColorMap();

	std::string objectDirName = cmdLineArgs.getString("wd");

	// 模型文件
	std::string inputFileName = objectDirName + "/build.in";
	std::string pmlSettingFileName = objectDirName + "/PML.in";
	std::string faceBndSettingFileName = objectDirName + "/FaceBnd.in";
	std::string speciesSettingFileName = objectDirName + "/Species.in";
	std::string globalSettingFileName = objectDirName + "/GlobalSetting.in";
	std::string fldsDgnSettingFileName = objectDirName + "/FieldsDgn.in";
	std::string staticFldsSettingFileName = objectDirName + "/StaticNodeFlds.in";
	std::string materialSettingFileName = objectDirName + "/MatData.in";

	std::string unitFileName = objectDirName + "/UnisSystem.h5";
	std::string globalGridFileName = objectDirName + "/ZRGrid.h5";
	std::string gridbndFileName = objectDirName + "/GridBndDatas.h5";

	std::string outputFileName = objectDirName + "/model.xml";
	std::string outputDirName = objectDirName + "/result";
	std::string outputFldsDgnName = outputDirName + "/FieldsDgn.txt";
	std::string ptclInformation = outputDirName + "/ptcl.txt";
	std::string outputFldPrefixName = outputDirName + "/CEMPIC";
	std::string outputNodeFldsDgnName = outputDirName + "/NodeFieldsDgn.txt";
	std::string outputPtclsDgnName = outputDirName + "/PtclsDgn.txt";
	std::string outputCurrentDgnName = outputDirName + "/CurrentDgn.txt";

	std::string dynEMKind = "SCFIT";
	std::string dynEMKind_Cyl3D = "SI_SC_Cyl3D";
	std::string dynEMKind_Matrix_Cyl3D = "SI_SC_Matrix_Cyl3D";

	if (access(outputDirName.c_str(), 0) == -1)
	{
		std::cout << "outputDirName : " << objectDirName.c_str() << std::endl;
		int flag = mkdir(outputDirName.c_str(), 0777);
		if (flag != 0)
		{
			std::cout << "output direction can not be make correctly" << std::endl;
		}
	}

	std::cout << "+ Start the simulation";
	std::cout << "  + BackGround Material Type";
	std::cout << "    - EMFREESPACE = " << EMFREESPACE << std::endl;
	std::cout << "    - PEC = " << PEC << std::endl;
	std::cout << "    - USERDEFINED = " << USERDEFINED << std::endl;

	// 读取build.in设置参数

	Standard_Integer unitScale = 0;
	Standard_Integer backGround = EMFREESPACE; // EMFREESPACE, PEC
	Standard_Real geomAlgoTol = 1.0e-10;
	Standard_Size dimPhi = 1;
	TxHierAttribSet tha = ReadAttrib(inputFileName.c_str(), "BuildSetting");

	if (tha.hasOption("backGround"))
	{
		backGround = tha.getOption("backGround");
	}
	if (tha.hasOption("unitScale"))
	{
		unitScale = tha.getOption("unitScale");
	}
	if (tha.hasParam("geomAlgoTol"))
	{
		geomAlgoTol = tha.getParam("geomAlgoTol");
	}

	UnitsSystemDef *theUnitsSystem = new UnitsSystemDef();
	theUnitsSystem->SetUnitScaleOfLength(unitScale);
	std::cout << "  + Parameter Setting in build.in" << std::endl;
	std::cout << "    + BackGround Material" << std::endl;
	std::cout << "      - Material Index = " << backGround << std::endl;
	std::cout << "    + UnitScale" << std::endl;
	std::cout << "      - UnitScale = " << unitScale << std::endl;
	std::cout << "    + geomAlgoTol" << std::endl;
	std::cout << "      - geomAlgoTol = " << geomAlgoTol << std::endl;

	std::cout << "  + Modle.xml Generation" << std::endl;
	Model_Ctrl *theModelCtrl = new Model_Ctrl();
	Geom_TxtBuilders *theAllBuilders = new Geom_TxtBuilders();
	OCAFDocumentCtrl *theDocCtrl = new OCAFDocumentCtrl();
	BeginReadOCCDocument();
	theDocCtrl->OnNewDocument();
	theAllBuilders->Init(theDocCtrl);
	theAllBuilders->SetAttrib(tha);
	theDocCtrl->OnSaveDocument(outputFileName);

	std::cout << "  + Modle Ctrl Generation" << std::endl;
	ReadOCCDocument(theDocCtrl, theModelCtrl);
	EndReadOCCDocument();
	theModelCtrl->Setup();

	delete theAllBuilders;
	delete theDocCtrl;

	std::cout << "  + Grid Generation" << std::endl;
	ZRGrid_Ctrl *theGridCtrl = new ZRGrid_Ctrl(theUnitsSystem, theModelCtrl);
	theGridCtrl->SetAttrib(tha);
	theGridCtrl->Build();
	theGridCtrl->SetupGrid();
	ZRGrid *theGG = theGridCtrl->GetZRGrid();
	ZRDefine *theGD = theGridCtrl->GetZRDefine();

	if (tha.hasOption("PhiDim"))
	{
		dimPhi = tha.getOption("PhiDim");
	}
	theGG->SetPhiNumber(dimPhi);

	int n_vertex_z = theGG->GetVertexDimension(0);
	int n_vertex_r = theGG->GetVertexDimension(1);
	int n_vertex_f = theGG->GetPhiNumber();
	std::cout << "    - n_vertex_z = " << n_vertex_z << std::endl;
	std::cout << "    - n_vertex_r = " << n_vertex_r << std::endl;
	std::cout << "    - n_vertex_f = " << n_vertex_f << std::endl;
	// exit(-1);

	std::cout << "  + Grid Bound Datas Generation" << std::endl;
	TxHierAttribSet theMatDataTha = ReadAttrib(materialSettingFileName.c_str(), "matdatasetting");
	Grid_Generation *theGridGeneration = new Grid_Generation(theGG, theGD, theModelCtrl, backGround, geomAlgoTol);
	theGridGeneration->BuildGridBndDatas();
	GridBndData *theGBData = theGridGeneration->GetGridBndDatas();
	if (theMatDataTha.getNumAttribs())
	{
		theGBData->SetAttrib(objectDirName, theMatDataTha);
	}

	// 读取PML.in

	std::cout << "  + Build PMLDataTool" << std::endl;
	TxHierAttribSet thePMLTha = ReadAttrib(pmlSettingFileName.c_str(), "PMLSetting");
	PMLDataDefine *thePMLTool = new PMLDataDefine();
	thePMLTool->SetAttrib(thePMLTha);
	theGG->ScaleAccordingUnitSystem(theUnitsSystem);
	GridGeometry *theGridGeom = new GridGeometry(theGG, theGBData);
	theGridGeom->SetPMLDataDefine(thePMLTool);
	theGridGeom->Setup();

	std::cout << "  + Build the 3D GridGeom Grid " << std::endl;
	GridGeometry_Cyl3D *theGridGeom_Cyl3D = new GridGeometry_Cyl3D(theGG, theGBData, dimPhi);
	theGridGeom_Cyl3D->SetPMLDataDefine(thePMLTool);
	theGridGeom_Cyl3D->Setup();
	theGridGeom_Cyl3D->Build_Near_Edge();
	std::cout << "      - End Of the 3D GridGeom3D Grid " << std::endl;

	std::cout << "  + Set Fields PhysDatas & Locate Memeory" << std::endl;
	ComboFieldsDefineRules *theComboFldDefRules = new ComboFieldsDefineRules();
	theComboFldDefRules->Setup_Fields_PhysDatasNum_AccordingMaterialDefine();
	// field_define_center 根据theComboFldDefRules中的规则对theGridGeom开辟物理数据
	ComboFields_DefineCntr *theFldsDefCntr = NULL;
	theFldsDefCntr = new ComboFields_DefineCntr(theGridGeom, theComboFldDefRules);
	theFldsDefCntr->LocateMemeory_For_FieldsPhysDatas();
	//////////////////Allocate Memory For the GridGeom3D////////////////////////////////
	theFldsDefCntr->SetGridGeom_Cyl3D(theGridGeom_Cyl3D);
	theFldsDefCntr->LocateMemeory_For_3DFieldsPhysDatas();

	Standard_Real dl = theGG->GetMinStep();
	Standard_Real dt = 0;

	std::cout << "  + Read Parameter from GlobalSetting.in" << std::endl;
	std::cout << "  + Global Para " << std::endl;
	Standard_Real cfl_scale = 1.0;
	Standard_Integer emAdvanceOrder = 1;
	Standard_Integer InterpolateType = 0;
	Standard_Real emDamping = 0.0;
	bool doDamping = false;
	Standard_Real T = 20.0e-9;

	TxHierAttribSet theGlobalTha = ReadAttrib(globalSettingFileName.c_str(), "globalsetting");
	if (theGlobalTha.hasParam("simulationTime"))
	{
		T = theGlobalTha.getParam("simulationTime");
	}

	if (theGlobalTha.hasParam("CFLScale"))
	{
		cfl_scale = theGlobalTha.getParam("CFLScale");
	}

	if (theGlobalTha.hasOption("emAdvanceOrder"))
	{
		emAdvanceOrder = theGlobalTha.getOption("emAdvanceOrder");
		std::cout << "    - emAdvanceOrder = " << emAdvanceOrder << std::endl;
	}

	if (theGlobalTha.hasString("dynEMkind"))
	{
		dynEMKind = theGlobalTha.getString("dynEMkind");
		std::cout << "    - dynEMKind = " << dynEMKind << std::endl;
	}

	if (theGlobalTha.hasParam("emDamping"))
	{
		emDamping = theGlobalTha.getParam("emDamping");
		if (emDamping > 0.0)
		{
			doDamping = true;
			std::cout << "    - do em damping" << std::endl;
		}
	}

	if (theGlobalTha.hasOption("InterpolateType"))
	{
		InterpolateType = theGlobalTha.getOption("InterpolateType");
		std::cout << "    - InterpolateType\t=\t" << InterpolateType << std::endl;
	}

	Standard_Size Index[2] = {0, 2};
	double dl_phi = theGG->GetCoordComp_From_VertexVectorIndx(1, Index) * 2 * mksConsts.pi / dimPhi / 1.0;
	if (dl_phi < dl)
		dl = dl_phi;
	ComputeCFLConstrain(dl, dt);
	dt = cfl_scale * dt;

	Standard_Integer nstep = Standard_Size(T / dt);

	std::cout << "    - dl = " << dl << std::endl;
	std::cout << "    - dt = " << dt << std::endl;
	std::cout << "    - nstep = " << nstep << std::endl;

	if (dimPhi >= 2)
	{
#ifdef __CUDA__
		Dynamic_ComboEMFieldsBase *theEMFields_Cyl3D = TxMakerMap<Dynamic_ComboEMFieldsBase>::getNew(dynEMKind_Matrix_Cyl3D);
#elif defined(__MATRIX__)
		Dynamic_ComboEMFieldsBase *theEMFields_Cyl3D = TxMakerMap<Dynamic_ComboEMFieldsBase>::getNew(dynEMKind_Matrix_Cyl3D);
#else
		Dynamic_ComboEMFieldsBase *theEMFields_Cyl3D = TxMakerMap<Dynamic_ComboEMFieldsBase>::getNew(dynEMKind_Cyl3D);
		// Dynamic_ComboEMFieldsBase* theEMFields_Cyl3D = TxMakerMap<Dynamic_ComboEMFieldsBase>::getNew(dynEMKind_Matrix_Cyl3D);
#endif

		if (theEMFields_Cyl3D == NULL)
		{
			std::cout << "\t Dynamic EM Fields Is not properly defined--------------I will exit" << std::endl;
			exit(1);
		}
		else
		{
			std::cout << "\t Dynamic EM Fields is properly defined--------------I will go on" << std::endl;
		}

		std::cout << "  + Start to Build The EMFields_Cyl3D" << std::endl;
		theEMFields_Cyl3D->SetFldsDefCntr(theFldsDefCntr);
		theEMFields_Cyl3D->SetDelTime(dt);
		theEMFields_Cyl3D->SetOrder(emAdvanceOrder);
		theEMFields_Cyl3D->SetDamping(emDamping);
		theEMFields_Cyl3D->Setup();
		theEMFields_Cyl3D->ZeroPhysDatas();
		std::cout << "     - The EMFields_Cyl3D is Built Successfully" << std::endl;

		std::cout << "  + Start to Set The Source 3D" << std::endl;
		TxHierAttribSet theFaceBndTha = ReadAttrib(faceBndSettingFileName.c_str(), "facesbndsetting");
		theEMFields_Cyl3D->InitFldSrcs_Cyl3D();
		theEMFields_Cyl3D->SetFldSrcsAttrib_Cyl3D(objectDirName, theFaceBndTha);
		std::cout << "     - The Source is Set Successfully \n"
				  << std::endl;

		if (!theEMFields_Cyl3D->IsPhysDataMemoryLocated())
		{
			std::cout << "     - PhysDataMemory Of the EMFields_Cyl3D  Is not properly located" << std::endl;
			return 0;
		}
		else
		{
			std::cout << "     - PhysDataMemory Of the EMFields_Cyl3D Is properly located" << std::endl;
		}

		TxHierAttribSet theFldsDgnTha = ReadAttrib(fldsDgnSettingFileName.c_str(), "FldsDgnSetting");
		FieldsDgnSets *theFldsDgnTool = new FieldsDgnSets(theFldsDefCntr);
		theFldsDgnTool->SetDelTime(dt);
		theFldsDgnTool->SetAttrib(theFldsDgnTha);

		ofstream DgnStream(outputFldsDgnName.c_str());
		ofstream PtclStream(ptclInformation.c_str());
		ofstream NodeFldsStream(outputNodeFldsDgnName.c_str());
		ofstream PtclDgnStream(outputPtclsDgnName.c_str());
		ofstream CurrentDgnStream(outputCurrentDgnName.c_str());
		theFldsDgnTool->DumpHead(DgnStream);

		std::cout << "  + Start to set Static Field." << std::endl;
		TxHierAttribSet staticflds_attrib = ReadAttrib(staticFldsSettingFileName.c_str(), "staticFldSetting");
		NodeField_Cyl3D node_field_Cyl3D(theGridGeom_Cyl3D, (SI_SC_ComboEMFields_Cyl3D *)theEMFields_Cyl3D, theComboFldDefRules);
		node_field_Cyl3D.setAttrib(staticflds_attrib);
		std::cout << "     - The NodeFields_Cyl3D is Set Successfully \n"
				  << std::endl;

		std::cout << "  + Start to Set the Species" << std::endl;
		TxHierAttribSet species_attrib = ReadAttrib(speciesSettingFileName.c_str(), "species");
		Species_Cyl3D the_species_Cyl3D(&node_field_Cyl3D, theGG, theGridGeom_Cyl3D);
		the_species_Cyl3D.setAttrib(species_attrib);
		std::cout << "     - The Species_Cyl3D is Set Successfully \n"
				  << std::endl;

		int emit_face_mask = the_species_Cyl3D.mask;
		double threshold = the_species_Cyl3D.threshold;
		double edgeEnhance = the_species_Cyl3D.edgeEnhance;
		CLEmit_Cyl3D cl_emit_Cyl3D(theGridGeom_Cyl3D, &node_field_Cyl3D, &the_species_Cyl3D, emit_face_mask, threshold, edgeEnhance);
		std::cout << "     - The CLEmit_Cyl3D is Set Successfully \n"
				  << std::endl;

		// 为TECIO创建数据
		NodeFldsTecio_Cyl3D *theNodefldsTecio_Cyl3D = new NodeFldsTecio_Cyl3D(theGridGeom_Cyl3D, theComboFldDefRules,
																			  theEMFields_Cyl3D, &node_field_Cyl3D);

		std::cout << "  + The Simulation starts !" << std::endl;
		double t = 0;

		CurrentTime();
		time_t startTime0;
		startTime0 = time(NULL);

		double iStart_Fields, iElaps_Fields;
		double iStart_Species, iElaps_Species;
		float time_elapsed_, time_elapsed_node;

		double iStart[8], iElaps[8], AllElaps[10] = {0.0};
		theEMFields_Cyl3D->InitMatrixData();

		double iStart0, iElaps0;
		iStart0 = seconds();

#ifdef __CUDA__
		theEMFields_Cyl3D->BuildCUDADatas();
		node_field_Cyl3D.BuildCUDADatas();
		the_species_Cyl3D.BuildCUDADatas();
		the_species_Cyl3D.Cuda_Init_Cuda_Constant_Vars();
#endif

		for (int i = 0; i <= nstep; i++)
		{
			t += dt;
			if (i % 100 == 0)
			{
				std::cout << " \n"
						  << std::endl;
				std::cout << "Total Steps and Current Steps: \t" << nstep << " \t" << i << std::endl;
#ifdef __CUDA__
				size_t ptcl_num = the_species_Cyl3D.count_ptcl_cuda();
#else
				size_t ptcl_num = the_species_Cyl3D.count_ptcl();
#endif
				std::cout << "step  = \t" << i << " \t" << ptcl_num << " ptcls exist----------------in GridGeom_3D" << std::endl;

				time_t elapsed = time(NULL);
				double elapsedTime = (double)(elapsed - startTime0);
				double remainTime = (double)(elapsedTime / (i + 1) * nstep);
				double pregressRate = elapsedTime / remainTime * 100;
				int elapsedTimeHour = floor(elapsedTime / 3600);
				elapsedTime = elapsedTime - 3600 * elapsedTimeHour;
				int elapsedTimeMin = floor(elapsedTime / 60);
				elapsedTime = elapsedTime - 60 * elapsedTimeMin;
				int elapsedTimeSecond = floor(elapsedTime);
				int remainTimeHour = floor(remainTime / 3600);
				remainTime = remainTime - 3600 * remainTimeHour;
				int remainTimeMin = floor(remainTime / 60);
				remainTime = remainTime - 60 * remainTimeMin;
				int remainTimeSecond = floor(remainTime);

				printf("dt [%2.2e s] Time [%2.2e s]  Step [%d / %d] Ptcl [%d] Progress [%02d:%02d:%02d/%02d:%02d:%02d] %3.2f%%\n", dt, t, i, nstep, ptcl_num,
					   elapsedTimeHour, elapsedTimeMin, elapsedTimeSecond, remainTimeHour, remainTimeMin, remainTimeSecond, pregressRate);
			}

			// 电磁场推进
			{
				iStart[0] = seconds();
				theEMFields_Cyl3D->AdvanceWithDamping(); // include paticle source
				iElaps[0] = seconds() - iStart[0];
				AllElaps[0] += iElaps[0];
			}

			// 节点场计算
			{
				iStart[1] = seconds();
#ifdef __CUDA__
				time_elapsed_node = node_field_Cyl3D.update_cuda();
#elif defined(__MATRIX__)
				node_field_Cyl3D.update_matrix();
#else
				node_field_Cyl3D.update();
#endif
				AllElaps[8] += time_elapsed_node;
				iElaps[1] = seconds() - iStart[1];
				AllElaps[1] += iElaps[1];
			}

			// 电荷置零
			{
				iStart[2] = seconds();
#ifdef __CUDA__
				node_field_Cyl3D.clear_current_density_cuda();
#elif defined(__MATRIX__)
				node_field_Cyl3D.clear_current_density();
#else
				node_field_Cyl3D.clear_current_density();
#endif
				iElaps[2] = seconds() - iStart[2];
				AllElaps[2] += iElaps[2];
			}

			// 粒子推进
			{
				iStart[3] = seconds();
#ifdef __CUDA__
				time_elapsed_ = the_species_Cyl3D.Advance_With_Cuda(dt);
#elif defined(__MATRIX__)
				the_species_Cyl3D.advance(dt);
#else
				the_species_Cyl3D.advance(dt);
#endif
				AllElaps[7] += time_elapsed_;
				iElaps[3] = seconds() - iStart[3];
				AllElaps[3] += iElaps[3];
			}

			// 粒子发射
			// 这一部分应该只改了接口, 尚未实现
			{
				iStart[4] = seconds();
#ifdef __CUDA__
				// cl_emit_Cyl3D.emit_cuda(dt);
				cl_emit_Cyl3D.emit_gauss_cuda(dt);
#else
				cl_emit_Cyl3D.emit_czg(dt);
				// cl_emit_Cyl3D.emit_gauss(dt);
#endif
				iElaps[4] = seconds() - iStart[4];
				AllElaps[4] += iElaps[4];
			}

			// 电流计算
			{
				iStart[5] = seconds();
#ifdef __CUDA__
				node_field_Cyl3D.step_to_conformal_current_test_cuda();
#elif defined(__MATRIX__)
				node_field_Cyl3D.step_to_conformal_current_test_matrix();
#else
				node_field_Cyl3D.step_to_conformal_current_test();
#endif
				iElaps[5] = seconds() - iStart[5];
				AllElaps[5] += iElaps[5];
			}

			// 数据诊断
			{
				iStart[6] = seconds();
#ifdef __CUDA__
				// theEMFields_Cyl3D->TransDgnData(); // 场数据传输开关
				// node_field_Cyl3D.TransDgnData();   // 节点场数据传输开关
#endif

				theFldsDgnTool->Advance();
				theFldsDgnTool->Dump(DgnStream);
				iElaps[6] = seconds() - iStart[6];
				AllElaps[6] += iElaps[6];
			}

			// node_field_Cyl3D.record_NodeField(NodeFldsStream);
			// node_field_Cyl3D.record_Current(CurrentDgnStream);

			// if(i == 3100){
			// 	the_species_Cyl3D.record_PtclInfo(PtclDgnStream);
			// }

			// if(i == (Standard_Size)(1.2e-9/dt))
			// {
			// 	// theNodefldsTecio_Cyl3D->Tecio_rphiFacet_Elec_Phi(outputDirName,0.005,1);
			// 	theNodefldsTecio_Cyl3D->Tecio_Elec_Phi(outputDirName);
			// }

			// if(i == (Standard_Size)(1.22e-9/dt))
			// {
			// 	theNodefldsTecio_Cyl3D->Tecio_rphiFacet_Elec_Phi(outputDirName,0.005,2);
			// }
			// if(i == (Standard_Size)(42.04e-9/dt))
			// {
			// 	theNodefldsTecio_Cyl3D->Tecio_rphiFacet_Elec_Phi(outputDirName,0.48,3);
			// }
			// if(i == (Standard_Size)(42.06e-9/dt))
			// {
			// 	theNodefldsTecio_Cyl3D->Tecio_rphiFacet_Elec_Phi(outputDirName,0.48,4);
			// }
		}

#ifdef __CUDA__
		theEMFields_Cyl3D->CleanCUDADatas();
		the_species_Cyl3D.CleanCUDADatas();
		node_field_Cyl3D.CleanCUDADatas();
#endif

		iElaps0 = seconds() - iStart0;
		printf("cpu reduce      elapsed %f ms \n", iElaps0 * 1000);

		printf("EMFields elapsed %f ms \n", AllElaps[0] * 1000);		 // 共形场推进花费时间
		printf("NodeField update elapsed %f ms \n", AllElaps[1] * 1000); // 节点场推进花费时间
		printf("Clear current elapsed %f ms \n", AllElaps[2] * 1000);	 // 清除电流电荷花费时间
		printf("Species elapsed %f ms \n", AllElaps[3] * 1000);			 // 粒子推进花费时间
		printf("Emit elapsed %f ms \n", AllElaps[4] * 1000);			 // 粒子发射花费时间
		printf("Current test elapsed %f ms \n", AllElaps[5] * 1000);	 // 电流写入边花费时间
		printf("Dgn elapsed %f ms \n", AllElaps[6] * 1000);				 // 数据诊断花费时间
		printf("Others elapsed %f ms \n", (iElaps0 - AllElaps[0] - AllElaps[1] - AllElaps[2] - AllElaps[3] - AllElaps[4] - AllElaps[5] - AllElaps[6]) * 1000);

		std::cout << "  The Simulation ends !" << std::endl;

		delete theEMFields_Cyl3D;
		delete theComboFldDefRules;
		delete theFldsDefCntr;
		delete theGridGeom;
		delete theGridGeom_Cyl3D;
		delete theGridGeneration;
		delete theGridCtrl;
		delete theModelCtrl;
		delete theUnitsSystem;
		delete theNodefldsTecio_Cyl3D;

		return 0;
	}
};

void ComputeCFLConstrain(const Standard_Real dl, Standard_Real &dt)
{
	dt = fabs(dl) / 3.0e8 / sqrt(3.0);
}

//=======================================================================
// function : ReadOCCStd
// purpose  :
//=======================================================================
Standard_Boolean BeginReadOCCDocument()
{
	SetupColorMap();
	return Standard_True;
	std::cout << "BeginReadOCCDocument-----------------------1" << std::endl;
}

//=======================================================================
// function : ReadOCCStd
// purpose  :
//=======================================================================
Standard_Boolean EndReadOCCDocument()
{
	ClearColorMap();
	return Standard_True;
}

//=======================================================================
// function : ReadOCCStd
// purpose  :
//=======================================================================
Standard_Boolean ReadOCCDocument(OCAFDocumentCtrl *&theDocumentCtrl, Model_Ctrl *&theModelCtrl)
{
	Standard_Boolean result = Standard_True;

	Handle(TDocStd_Document) theOCAFDoc = theDocumentCtrl->GetOCAFDoc();

	if (theOCAFDoc.IsNull())
		return result;

	TDF_Label theRootLabel = theOCAFDoc->GetData()->Root();
	if (!theRootLabel.IsNull())
	{
		TDF_ChildIterator anIterator(theRootLabel);

		for (; anIterator.More(); anIterator.Next())
		{
			TDF_Label theLabel = anIterator.Value();
			Handle(TDataStd_TreeNode) aNode;

			if (!theLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode))
			{
				result = Standard_False;
				break;
			}

			if (OCAF_ObjectTool::IsOneObject(aNode))
			{
				OCAF_Object anInterface(aNode);
				if (anInterface.GetObjectMask() > 0)
				{
					TopoDS_Shape theShape = anInterface.GetObjectValue();
					Standard_Integer theMaterialIndex = anInterface.GetObjResultMaterial();
					Standard_Boolean IsOK = Standard_False;
					Standard_Integer theMaterialType = OCAF_ColorMap::GetMaterialType(theMaterialIndex, IsOK);
					Standard_Integer theMask = anInterface.GetObjectMask();

					if (theShape.ShapeType() == TopAbs_SOLID)
					{
						theModelCtrl->AppendShape(theShape, theMaterialType);
						theModelCtrl->SetShapeMask(theShape, theMask);

#ifdef READSTD_DBG
						std::cout << "The material type index of this Shape is\t=\t" << theMaterialIndex << std::endl;
						std::cout << "The material type of this Shape is\t=\t" << theMaterialType << std::endl;
#endif
					}
				}
			}
		}

		TDF_ChildIterator anFaceIterator(theRootLabel);
		for (; anFaceIterator.More(); anFaceIterator.Next())
		{
			TDF_Label theLabel = anFaceIterator.Value();
			Handle(TDataStd_TreeNode) aNode;
			if (!theLabel.FindAttribute(TDataStd_TreeNode::GetDefaultTreeID(), aNode))
			{
				result = Standard_False;
				break;
			}

			if (OCAF_ObjectTool::IsOneObject(aNode))
			{
				OCAF_Object anInterface(aNode);
				if (anInterface.GetObjectMask() > 0)
				{
					TopoDS_Shape theShape = anInterface.GetObjectValue();

					Standard_Integer theMaterialIndex = anInterface.GetObjResultMaterial();
					Standard_Boolean IsOK = Standard_False;
					Standard_Integer theMaterialType = OCAF_ColorMap::GetMaterialType(theMaterialIndex, IsOK);
					Standard_Integer theMask = anInterface.GetObjectMask();

					if (theShape.ShapeType() == TopAbs_FACE)
					{
						const TopoDS_Face &theFace = TopoDS::Face(theShape);
						theModelCtrl->AppendSpecialFace(theFace, theMaterialType);
						theModelCtrl->SetSpecialFaceMask(theFace, theMask);

#ifdef READSTD_DBG
						std::cout << "The material type index of this Face is\t" << theMaterialIndex << std::endl;
						std::cout << "The material type of this Face is\t" << theMaterialType << std::endl;
#endif
					}
				}
			}
		}
	}

	return result;
}
