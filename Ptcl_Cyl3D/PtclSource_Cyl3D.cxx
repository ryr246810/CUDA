#include <PtclSource_Cyl3D.cuh>
#include <Species_Cyl3D.cuh>
#include <IndexAndWeights_Cyl3D.cuh>
#include <intersect_func_Cyl3D.cuh>

// int emit_cnt;

void BeamEmit_Cyl3D::
    setup()
{
    double q = species->charge;
    double m = species->mass;
    double vtmp;
    phi_To_v(q, m, m_voltage, vtmp); // move velcocity
    v_to_u(vtmp, m_u);               // relativsic velocity
    m_gamma = m_u / vtmp;
}

void CLEmit_Cyl3D::
    emit_cuda(double dt)
{
    double q2m = species->chargeOverMass;
    double macro_charge = species->macroCharge;
    double q2dt = macro_charge / dt;
    double dl = global_grid->GetMinStep();
    int n_segment;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    TxVector<double> probe_pnt, ptcl_pos;
    TxVector<double> E_probe, J_probe, vel;
    IndexAndWeights_Cyl3D iw_start, iw_end;
    double const1 = q2m * mksConsts.epsilon0 * mksConsts.epsilon0 / (3.0 * dl);
    TxVector<double> eField;
    TxVector<double> randomDis_L;

    vector<IndexAndWeights_Cyl3D> iw_segment_t;
    vector<double> q2dt_t;
    vector<TxVector<double>> frac_disp_t;
    int n_elem = emit_elements.size();
    for (int i = 0; i < n_elem; i++)
    {
        TxVector<double> emit_dir = emit_elements[i]->emit_dir;
        TxVector<double> emit_pnt = emit_elements[i]->emit_pnt;
        double area = emit_elements[i]->emit_area;

        global_grid->ComputeIndexVecAndWeightsInGrid(emit_pnt, iw_start.indx, iw_start.wl, iw_start.wu);
        ComputeEmissionElecField_CUDA(emit_elements[i], probe_pnt, randomDis_L, eField);

        // double E_normal = eField[1]; // transform to 1D vision
        double E_normal = eField.Dot(emit_dir);
        if (emit_elements[i]->edgeEnhance)
            E_normal *= m_edgehance;
        probe_pnt = probe_pnt - randomDis_L * 0.5;

        node_field->fill_with_J_CUDA(probe_pnt, J_probe);

        double I_probe = J_probe.Dot(emit_dir);
        double I_need = E_normal * sqrt(fabs(const1 * E_normal)) * area;

        double num_ptcl = (I_need - I_probe) * dt / macro_charge;

        int num_emit;
        num_emit = (int)(num_ptcl);
        double remain_wt = num_ptcl - num_emit;

        if (fabs(E_normal) < m_threshold)
        {
            // continue;
        }

        for (int i_ptcl = 0; i_ptcl < num_emit; i_ptcl++)
        {
            TxVector<double> disp;
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = emit_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            // emit_elements[i]->get_random_tan_dist(disp);
            disp += emit_dir * (vtmp * emit_time * 0.5);
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                double cell_vol = node_field->get_cell_volume(i_cell, j_cell, k_cell);
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> current_density = disp_3d * (fraction[i_seg] * q2dt / cell_vol);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                // node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt);
                iw_segment_t.push_back(iw_segment[i_seg]);
                frac_disp_t.push_back(frac_disp);
                q2dt_t.push_back(q2dt);
                //*****************************************************************************************************************
                // node_field->accumulate_I(iw_segment[i_seg], frac_disp * 3.48, q2dt);
            }
            {
                // species->add_ptcl(ptcl_pos, iw_end, vel, 0);
                species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, 1.0);
            }
            // node_field->accumulate_Rho(iw_end,macro_charge);// tzh
        }

        if (remain_wt > 0.01)
        {
            TxVector<double> disp;
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = emit_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            // emit_elements[i]->get_random_tan_dist(disp);
            disp += emit_dir * (vtmp * emit_time * 0.5);
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                double cell_vol = node_field->get_cell_volume(i_cell, j_cell, k_cell);
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> current_density = disp_3d * (fraction[i_seg] * q2dt / cell_vol);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                // node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt*remain_wt);
                iw_segment_t.push_back(iw_segment[i_seg]);
                frac_disp_t.push_back(frac_disp);
                q2dt_t.push_back(q2dt * remain_wt);

                //*****************************************************************************************************************
            }
            {
                // species->add_ptcl(ptcl_pos, iw_end, vel, 0,remain_wt);
                species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, remain_wt);
            }
            // node_field->accumulate_Rho(iw_end,remain_wt*macro_charge);// tzh
        }
    }
    node_field->accumulate_I_CUDA(iw_segment_t, frac_disp_t, q2dt_t);
}

void CLEmit_Cyl3D::
    emit_czg(double dt)
{
    // exit(-1);
    double q2m = species->chargeOverMass;
    double macro_charge = species->macroCharge;
    double q2dt = macro_charge / dt;
    double dl = global_grid->GetMinStep();
    int n_segment;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    TxVector<double> probe_pnt, ptcl_pos;
    TxVector<double> E_probe, J_probe, vel;
    IndexAndWeights_Cyl3D iw_start, iw_end;
    double const1 = q2m * mksConsts.epsilon0 * mksConsts.epsilon0 / (3.0 * dl);
    TxVector<double> eField;
    TxVector<double> randomDis_L;

    vector<IndexAndWeights_Cyl3D> iw_segment_t;
    vector<double> q2dt_t;
    vector<TxVector<double>> frac_disp_t;
    int n_elem = emit_elements.size();
    for (int i = 0; i < n_elem; i++)
    {
        TxVector<double> emit_dir = emit_elements[i]->emit_dir;
        TxVector<double> emit_pnt = emit_elements[i]->emit_pnt;
        double area = emit_elements[i]->emit_area;

        global_grid->ComputeIndexVecAndWeightsInGrid(emit_pnt, iw_start.indx, iw_start.wl, iw_start.wu);
        ComputeEmissionElecField(emit_elements[i], probe_pnt, randomDis_L, eField);

        // double E_normal = eField[1]; // transform to 1D vision
        double E_normal = eField.Dot(emit_dir);
        if (emit_elements[i]->edgeEnhance)
            E_normal *= m_edgehance;
        probe_pnt = probe_pnt - randomDis_L * 0.5;

        node_field->fill_with_J(probe_pnt, J_probe);

        double I_probe = J_probe.Dot(emit_dir);
        double I_need = E_normal * sqrt(fabs(const1 * E_normal)) * area;

        double num_ptcl = (I_need - I_probe) * dt / macro_charge;

        int num_emit;
        num_emit = (int)(num_ptcl);
        double remain_wt = num_ptcl - num_emit;

        if (fabs(E_normal) < m_threshold)
        {
            // continue;
        }

        for (int i_ptcl = 0; i_ptcl < num_emit; i_ptcl++)
        {
            TxVector<double> disp;
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = emit_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            // emit_elements[i]->get_random_tan_dist(disp);
            disp += emit_dir * (vtmp * emit_time * 0.5);
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                double cell_vol = node_field->get_cell_volume(i_cell, j_cell, k_cell);
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> current_density = disp_3d * (fraction[i_seg] * q2dt / cell_vol);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                // node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt);
                iw_segment_t.push_back(iw_segment[i_seg]);
                frac_disp_t.push_back(frac_disp);
                q2dt_t.push_back(q2dt);
                //*****************************************************************************************************************
                // node_field->accumulate_I(iw_segment[i_seg], frac_disp * 3.48, q2dt);
            }
            {
                species->add_ptcl(ptcl_pos, iw_end, vel, 0);
                // species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, 1.0);
            }
            // node_field->accumulate_Rho(iw_end,macro_charge);// tzh
        }

        if (remain_wt > 0.01)
        {
            TxVector<double> disp;
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = emit_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            // emit_elements[i]->get_random_tan_dist(disp);
            disp += emit_dir * (vtmp * emit_time * 0.5);
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                double cell_vol = node_field->get_cell_volume(i_cell, j_cell, k_cell);
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> current_density = disp_3d * (fraction[i_seg] * q2dt / cell_vol);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                // node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt*remain_wt);
                iw_segment_t.push_back(iw_segment[i_seg]);
                frac_disp_t.push_back(frac_disp);
                q2dt_t.push_back(q2dt * remain_wt);

                //*****************************************************************************************************************
            }
            {
                species->add_ptcl(ptcl_pos, iw_end, vel, 0, remain_wt);
                // species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, remain_wt);
            }
            // node_field->accumulate_Rho(iw_end,remain_wt*macro_charge);// tzh
        }
    }
    node_field->accumulate_I(iw_segment_t, frac_disp_t, q2dt_t);
}

void CLEmit_Cyl3D::
    emit_gauss(double dt)
{
    double q2m = species->chargeOverMass;
    double macro_charge = species->macroCharge;
    double qType = fabs(macro_charge) / macro_charge;
    double q2dt = macro_charge / dt;
    double dl = global_grid->GetMinStep();
    int n_segment;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    TxVector<double> probe_pnt, ptcl_pos;
    TxVector<double> E_probe, vel, E_emit;
    double Rho_probe;
    TxVector2D<double> tmpDL;
    TxVector<double> randomDis_L;
    TxVector<double> eField;
    IndexAndWeights_Cyl3D iw_start, iw_end;

    int n_elem = emit_elements.size();
    for (int i = 0; i < n_elem; i++)
    {
        TxVector<double> emit_dir = emit_elements[i]->emit_dir;
        TxVector<double> emit_pnt = emit_elements[i]->emit_pnt;
        TxVector<double> normal_dir = emit_dir;
        double area = emit_elements[i]->emit_area;

        TxVector<double> mid_pnt = emit_elements[i]->mid_pnt;
        global_grid->ComputeIndexVecAndWeightsInGrid(emit_pnt, iw_start.indx, iw_start.wl, iw_start.wu);
        Standard_Size index[2];
        index[0] = iw_start.indx[0];
        index[1] = iw_start.indx[1];
        tmpDL = global_grid->GetSteps(index);
        if (fabs(emit_dir[0]) > 0.9)
            dl = tmpDL[0];
        else
            dl = tmpDL[1];

        TxVector<double> tmp_E;
        TxVector<double> probe_pnt;
        E_probe = TxVector<double>(0, 0, 0);
        Rho_probe = 0.0;
        double tmp_Rho;
        randomDis_L = (mid_pnt + emit_dir * dl - emit_pnt) * 0.5;
        probe_pnt = emit_pnt + randomDis_L;
        node_field->fill_with_E(probe_pnt, E_probe);
        node_field->fill_with_E(emit_pnt, E_emit);
        // ComputeEmissionElecField_gauss(emit_elements[i],probe_pnt,randomDis_L,E_probe);
        probe_pnt += randomDis_L * (-0.5);
        // node_field->fill_with_Rho(emit_pnt, Rho_probe);
        node_field->fill_with_Rho(probe_pnt, Rho_probe);

        double E_probe_scalar = E_probe.Dot(normal_dir); // transform to 1D vision
        double E_normal = E_emit.Dot(normal_dir);
        if (emit_elements[i]->edgeEnhance)
            E_probe_scalar *= m_edgehance;
        double emitVol = (probe_pnt[0] - emit_pnt[0]) * area;
        double Rho_multi = 1.0 * dl / (dl - (mid_pnt[0] - emit_pnt[0]));
        emitVol = Rho_multi * dl * area;
        double Charge_req = E_probe_scalar * area * mksConsts.epsilon0;

        double Charge_probe = Rho_probe * emitVol;
        double Charge_need = Charge_req - Charge_probe;
        if ((Charge_need)*qType < 0.0)
            continue;

        double num_ptcl = Charge_need / macro_charge;

        size_t num_emit;
        num_emit = (int)(num_ptcl);
        double remain_wt = num_ptcl - num_emit;

        if (E_probe_scalar != 0)
        {
            double emitEFieldDir;
            emitEFieldDir = fabs(E_probe_scalar) / (E_probe_scalar);
            if (fabs(E_probe_scalar) < m_threshold || emitEFieldDir * qType < 0.0)
            {
                // continue;
            }
        }

        double emitDt = dt;
        if (num_emit >= 1)
            emitDt = dt / ((double)(num_emit));
        for (int i_ptcl = 0; i_ptcl < num_emit; i_ptcl++)
        {
            TxVector<double> disp, disp1;
            // double emit_time = dt*randomFunc();
            // double emit_time = dt*(i_ptcl+1)/num_emit;
            // double emit_time = emitDt*(i_ptcl+1);
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = normal_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            emit_elements[i]->get_random_tan_dist(disp1);
            // emit_elements[i]->get_init_emit_dist(disp);
            // disp *=randomFunc();
            disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt);
                //*****************************************************************************************************************
            }
            species->add_ptcl(ptcl_pos, iw_end, vel, 0);

            // node_field->accumulate_Rho(iw_end,macro_charge);// tzh
        }
        if (remain_wt > 0.01)
        {
            TxVector<double> disp, disp1;
            // double emit_time = dt*randomFunc();
            // double emit_time = emitDt;
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = normal_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            emit_elements[i]->get_random_tan_dist(disp1);
            // emit_elements[i]->get_init_emit_dist(disp);
            // disp = disp*randomFunc();
            disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt * remain_wt);
                //*****************************************************************************************************************
            }
            species->add_ptcl(ptcl_pos, iw_end, vel, 0, remain_wt);
            // node_field->accumulate_Rho(iw_end,macro_charge);// tzh
        }
    }
}

void CLEmit_Cyl3D::
    emit_gauss_cuda(double dt)
{
    double q2m = species->chargeOverMass;
    double macro_charge = species->macroCharge;
    double qType = fabs(macro_charge) / macro_charge;
    double q2dt = macro_charge / dt;
    double dl = global_grid->GetMinStep();
    int n_segment;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    TxVector<double> probe_pnt, ptcl_pos;
    TxVector<double> E_probe, vel, E_emit;
    double Rho_probe;
    TxVector2D<double> tmpDL;
    TxVector<double> randomDis_L;
    TxVector<double> eField;
    IndexAndWeights_Cyl3D iw_start, iw_end;

    int n_elem = emit_elements.size();
    for (int i = 0; i < n_elem; i++)
    {
        TxVector<double> emit_dir = emit_elements[i]->emit_dir;
        TxVector<double> emit_pnt = emit_elements[i]->emit_pnt;
        TxVector<double> normal_dir = emit_dir;
        double area = emit_elements[i]->emit_area;

        TxVector<double> mid_pnt = emit_elements[i]->mid_pnt;
        global_grid->ComputeIndexVecAndWeightsInGrid(emit_pnt, iw_start.indx, iw_start.wl, iw_start.wu);
        Standard_Size index[2];
        index[0] = iw_start.indx[0];
        index[1] = iw_start.indx[1];
        tmpDL = global_grid->GetSteps(index);
        if (fabs(emit_dir[0]) > 0.9)
            dl = tmpDL[0];
        else
            dl = tmpDL[1];

        TxVector<double> tmp_E;
        TxVector<double> probe_pnt;
        E_probe = TxVector<double>(0, 0, 0);
        Rho_probe = 0.0;
        double tmp_Rho;
        randomDis_L = (mid_pnt + emit_dir * dl - emit_pnt) * 0.5;
        probe_pnt = emit_pnt + randomDis_L;
        node_field->fill_with_E_CUDA(probe_pnt, E_probe);
        node_field->fill_with_E_CUDA(emit_pnt, E_emit);
        // ComputeEmissionElecField_gauss(emit_elements[i],probe_pnt,randomDis_L,E_probe);
        probe_pnt += randomDis_L * (-0.5);
        // node_field->fill_with_Rho(emit_pnt, Rho_probe);
        node_field->fill_with_Rho_CUDA(probe_pnt, Rho_probe);

        double E_probe_scalar = E_probe.Dot(normal_dir); // transform to 1D vision
        double E_normal = E_emit.Dot(normal_dir);
        if (emit_elements[i]->edgeEnhance)
            E_probe_scalar *= m_edgehance;
        double emitVol = (probe_pnt[0] - emit_pnt[0]) * area;
        double Rho_multi = 1.0 * dl / (dl - (mid_pnt[0] - emit_pnt[0]));
        emitVol = Rho_multi * dl * area;
        double Charge_req = E_probe_scalar * area * mksConsts.epsilon0;

        double Charge_probe = Rho_probe * emitVol;
        double Charge_need = Charge_req - Charge_probe;
        if ((Charge_need)*qType < 0.0)
            continue;

        double num_ptcl = Charge_need / macro_charge;

        size_t num_emit;
        num_emit = (int)(num_ptcl);
        double remain_wt = num_ptcl - num_emit;

        if (E_probe_scalar != 0)
        {
            double emitEFieldDir;
            emitEFieldDir = fabs(E_probe_scalar) / (E_probe_scalar);
            if (fabs(E_probe_scalar) < m_threshold || emitEFieldDir * qType < 0.0)
            {
                // continue;
            }
        }

        double emitDt = dt;
        if (num_emit >= 1)
            emitDt = dt / ((double)(num_emit));
        for (int i_ptcl = 0; i_ptcl < num_emit; i_ptcl++)
        {
            TxVector<double> disp, disp1;
            // double emit_time = dt*randomFunc();
            // double emit_time = dt*(i_ptcl+1)/num_emit;
            // double emit_time = emitDt*(i_ptcl+1);
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = normal_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            emit_elements[i]->get_random_tan_dist(disp1);
            // emit_elements[i]->get_init_emit_dist(disp);
            // disp *=randomFunc();
            disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                node_field->accumulate_I_CUDA(iw_segment[i_seg], frac_disp, q2dt);
                //*****************************************************************************************************************
            }
            // species->add_ptcl(ptcl_pos, iw_end, vel, 0);
            species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, 0);

            // node_field->accumulate_Rho(iw_end,macro_charge);// tzh
        }
        if (remain_wt > 0.01)
        {
            TxVector<double> disp, disp1;
            // double emit_time = dt*randomFunc();
            // double emit_time = emitDt;
            double emit_time = dt * randomFunc();
            double utmp = fabs(q2m * E_normal * emit_time);
            double vtmp;
            u_to_v(utmp, vtmp);
            vel = normal_dir * utmp;
            emit_elements[i]->get_random_emit_dist(disp);
            emit_elements[i]->get_random_tan_dist(disp1);
            // emit_elements[i]->get_init_emit_dist(disp);
            // disp = disp*randomFunc();
            disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
            ptcl_pos = emit_pnt + disp;

            global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
            n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
            for (int i_seg = 0; i_seg < n_segment; i_seg++)
            {
                int i_cell = iw_segment[i_seg].indx[0];
                int j_cell = iw_segment[i_seg].indx[1];
                int k_cell = iw_segment[i_seg].indx[2];
                TxVector<double> disp_3d(disp[0], disp[1], 0);
                TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                //*****************************************************************************************************************
                node_field->accumulate_I_CUDA(iw_segment[i_seg], frac_disp, q2dt * remain_wt);
                //*****************************************************************************************************************
            }
            // species->add_ptcl(ptcl_pos, iw_end, vel, 0,remain_wt);
            species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, remain_wt);
            // node_field->accumulate_Rho(iw_end,macro_charge);// tzh
        }
    }
}

void CLEmit_Cyl3D::
    emit_gauss1(double dt)
{
    double q2m = species->chargeOverMass;
    double macro_charge = species->macroCharge;
    double qType = fabs(macro_charge) / macro_charge;
    double q2dt = macro_charge / dt;
    double dl = global_grid->GetMinStep();
    int n_segment;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    TxVector<double> probe_pnt, ptcl_pos;
    TxVector<double> E_probe, vel, E_emit;
    double Rho_probe;
    TxVector2D<double> tmpDL;
    TxVector<double> randomDis_L;
    TxVector<double> eField;
    IndexAndWeights_Cyl3D iw_start, iw_end;
    double reqCharge = 0;
    int n_elem = emit_elements.size();
    for (int i = 0; i < n_elem; i++)
    {
        TxVector<double> emit_dir = emit_elements[i]->emit_dir;
        TxVector<double> emit_pnt = emit_elements[i]->emit_pnt;
        TxVector<double> normal_dir = emit_dir;
        double area = emit_elements[i]->emit_area;

        TxVector<double> mid_pnt = emit_elements[i]->mid_pnt;
        global_grid->ComputeIndexVecAndWeightsInGrid(emit_pnt, iw_start.indx, iw_start.wl, iw_start.wu);
        double dz = global_grid->GetStep(0, iw_start.indx[0]);
        double dr = global_grid->GetStep(1, iw_start.indx[1]);
        TxVector<double> dl = TxVector<double>(dz, dr, 0);

        double emitLength;
        emitLength = fabs(dl.Dot(normal_dir));
        probe_pnt = mid_pnt + normal_dir * emitLength;
        node_field->fill_with_E(probe_pnt, E_probe);
        if (E_probe.Dot(normal_dir) == 0)
            continue;

        double emitEFieldDir = fabs(E_probe.Dot(normal_dir)) / (E_probe.Dot(normal_dir));
        double emitEField = fabs(E_probe.Dot(normal_dir)) * m_edgehance;
        double emitArea = emit_elements[i]->emit_area;
        double emitVol = emitLength * emitArea;

        if ((emitEField > m_threshold) || emit_elements[i]->emit_flag == 1)
            if (emitEFieldDir * qType > 0)
            {
                emit_elements[i]->emit_flag = 1;
                reqCharge = mksConsts.epsilon0 * emitEField * emitEFieldDir * emitArea;

                double trise = 2e-9;
                double curTime = node_field->EMFields->GetCurTime();
                reqCharge = reqCharge * (1 - pow((exp(-curTime / trise)), 2));

                probe_pnt = probe_pnt - normal_dir * emitLength * 0.5;
                node_field->fill_with_Rho(probe_pnt, Rho_probe);
                double Charge_probe = Rho_probe * emitVol;
                if (Charge_probe * qType < 0.0)
                {
                    Charge_probe = 0.0;
                }
                double neededCharge = reqCharge - Charge_probe;

                if ((neededCharge)*qType < 0.0)
                    continue;

                double wt = 1.0;
                double remainedWt = 0;

                int emitMacroPtclNum = 0;
                // int emitMacroPtclNum  = 3;
                if (emitMacroPtclNum > 0)
                {
                    double macro_charge_cur = fabs(neededCharge / emitMacroPtclNum);
                    wt = fabs(macro_charge_cur / macro_charge);
                }
                else
                {
                    double dnum = fabs(neededCharge / macro_charge);
                    emitMacroPtclNum = (int)(dnum);
                    remainedWt = dnum - emitMacroPtclNum;
                }
                for (int i_ptcl = 0; i_ptcl < emitMacroPtclNum; i_ptcl++)
                {
                    TxVector<double> disp, disp1;
                    double emit_time = dt * randomFunc();
                    double utmp = fabs(q2m * emitEField * emit_time);
                    double vtmp;
                    u_to_v(utmp, vtmp);
                    vel = normal_dir * utmp;
                    emit_elements[i]->get_random_emit_dist(disp);
                    emit_elements[i]->get_random_tan_dist(disp1);
                    disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
                    ptcl_pos = emit_pnt + disp;

                    global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
                    n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
                    for (int i_seg = 0; i_seg < n_segment; i_seg++)
                    {
                        int i_cell = iw_segment[i_seg].indx[0];
                        int j_cell = iw_segment[i_seg].indx[1];
                        int k_cell = iw_segment[i_seg].indx[2];
                        TxVector<double> disp_3d(disp[0], disp[1], 0);
                        TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                        //*****************************************************************************************************************
                        node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt);
                        //*****************************************************************************************************************
                    }
                    // species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, wt);
                    species->add_ptcl(ptcl_pos, iw_end, vel, 0, wt);
                }
                if (remainedWt > 0.01)
                {
                    TxVector<double> disp, disp1;
                    double emit_time = dt * randomFunc();
                    double utmp = fabs(q2m * emitEField * emit_time);
                    double vtmp;
                    u_to_v(utmp, vtmp);
                    vel = normal_dir * utmp;
                    emit_elements[i]->get_random_emit_dist(disp);
                    emit_elements[i]->get_random_tan_dist(disp1);
                    disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
                    ptcl_pos = emit_pnt + disp;

                    global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
                    n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
                    for (int i_seg = 0; i_seg < n_segment; i_seg++)
                    {
                        int i_cell = iw_segment[i_seg].indx[0];
                        int j_cell = iw_segment[i_seg].indx[1];
                        int k_cell = iw_segment[i_seg].indx[2];
                        TxVector<double> disp_3d(disp[0], disp[1], 0);
                        TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                        //*****************************************************************************************************************
                        node_field->accumulate_I(iw_segment[i_seg], frac_disp, q2dt * remainedWt);
                        //*****************************************************************************************************************
                    }
                    // species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, remainedWt);
                    species->add_ptcl(ptcl_pos, iw_end, vel, 0, remainedWt);
                }
            }
    }
}

void CLEmit_Cyl3D::
    emit_gauss_cuda1(double dt)
{
    double q2m = species->chargeOverMass;
    double macro_charge = species->macroCharge;
    double qType = fabs(macro_charge) / macro_charge;
    double q2dt = macro_charge / dt;
    double dl = global_grid->GetMinStep();
    int n_segment;
    double fraction[4];
    IndexAndWeights_Cyl3D iw_segment[4];
    TxVector<double> probe_pnt, ptcl_pos;
    TxVector<double> E_probe, vel, E_emit;
    double Rho_probe;
    TxVector2D<double> tmpDL;
    TxVector<double> randomDis_L;
    TxVector<double> eField;
    IndexAndWeights_Cyl3D iw_start, iw_end;
    double reqCharge = 0;
    int n_elem = emit_elements.size();
    for (int i = 0; i < n_elem; i++)
    {
        TxVector<double> emit_dir = emit_elements[i]->emit_dir;
        TxVector<double> emit_pnt = emit_elements[i]->emit_pnt;
        TxVector<double> normal_dir = emit_dir;
        double area = emit_elements[i]->emit_area;

        TxVector<double> mid_pnt = emit_elements[i]->mid_pnt;
        global_grid->ComputeIndexVecAndWeightsInGrid(emit_pnt, iw_start.indx, iw_start.wl, iw_start.wu);
        double dz = global_grid->GetStep(0, iw_start.indx[0]);
        double dr = global_grid->GetStep(1, iw_start.indx[1]);
        TxVector<double> dl = TxVector<double>(dz, dr, 0);

        double emitLength;
        emitLength = fabs(dl.Dot(normal_dir));
        probe_pnt = mid_pnt + normal_dir * emitLength;
        node_field->fill_with_E_CUDA(probe_pnt, E_probe);
        if (E_probe.Dot(normal_dir) == 0)
            continue;

        double emitEFieldDir = fabs(E_probe.Dot(normal_dir)) / (E_probe.Dot(normal_dir));
        double emitEField = fabs(E_probe.Dot(normal_dir)) * m_edgehance;
        double emitArea = emit_elements[i]->emit_area;
        double emitVol = emitLength * emitArea;

        if ((emitEField > m_threshold) || emit_elements[i]->emit_flag == 1)
            if (emitEFieldDir * qType > 0)
            {
                emit_elements[i]->emit_flag = 1;
                reqCharge = mksConsts.epsilon0 * emitEField * emitEFieldDir * emitArea;

                double trise = 2e-9;
                double curTime = node_field->EMFields->GetCurTime();
                reqCharge = reqCharge * (1 - pow((exp(-curTime / trise)), 2));

                probe_pnt = probe_pnt - normal_dir * emitLength * 0.5;
                node_field->fill_with_Rho_CUDA(probe_pnt, Rho_probe);
                double Charge_probe = Rho_probe * emitVol;
                if (Charge_probe * qType < 0.0)
                {
                    Charge_probe = 0.0;
                }
                double neededCharge = reqCharge - Charge_probe;

                if ((neededCharge)*qType < 0.0)
                    continue;

                double wt = 1.0;
                double remainedWt = 0;

                int emitMacroPtclNum = 0;
                // int emitMacroPtclNum  = 3;
                if (emitMacroPtclNum > 0)
                {
                    double macro_charge_cur = fabs(neededCharge / emitMacroPtclNum);
                    wt = fabs(macro_charge_cur / macro_charge);
                }
                else
                {
                    double dnum = fabs(neededCharge / macro_charge);
                    emitMacroPtclNum = (int)(dnum);
                    remainedWt = dnum - emitMacroPtclNum;
                }
                for (int i_ptcl = 0; i_ptcl < emitMacroPtclNum; i_ptcl++)
                {
                    TxVector<double> disp, disp1;
                    double emit_time = dt * randomFunc();
                    double utmp = fabs(q2m * emitEField * emit_time);
                    double vtmp;
                    u_to_v(utmp, vtmp);
                    vel = normal_dir * utmp;
                    emit_elements[i]->get_random_emit_dist(disp);
                    emit_elements[i]->get_random_tan_dist(disp1);
                    disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
                    ptcl_pos = emit_pnt + disp;

                    global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
                    n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
                    for (int i_seg = 0; i_seg < n_segment; i_seg++)
                    {
                        int i_cell = iw_segment[i_seg].indx[0];
                        int j_cell = iw_segment[i_seg].indx[1];
                        int k_cell = iw_segment[i_seg].indx[2];
                        TxVector<double> disp_3d(disp[0], disp[1], 0);
                        TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                        //*****************************************************************************************************************
                        node_field->accumulate_I_CUDA(iw_segment[i_seg], frac_disp, q2dt);
                        //*****************************************************************************************************************
                    }
                    species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, wt);
                }
                if (remainedWt > 0.01)
                {
                    TxVector<double> disp, disp1;
                    double emit_time = dt * randomFunc();
                    double utmp = fabs(q2m * emitEField * emit_time);
                    double vtmp;
                    u_to_v(utmp, vtmp);
                    vel = normal_dir * utmp;
                    emit_elements[i]->get_random_emit_dist(disp);
                    emit_elements[i]->get_random_tan_dist(disp1);
                    disp += emit_dir * (vtmp * emit_time * 0.5) + disp1;
                    ptcl_pos = emit_pnt + disp;

                    global_grid->ComputeIndexVecAndWeightsInGrid(ptcl_pos, iw_end.indx, iw_end.wl, iw_end.wu);
                    n_segment = species->frac_segment(iw_start, iw_end, emit_pnt, disp, fraction, iw_segment);
                    for (int i_seg = 0; i_seg < n_segment; i_seg++)
                    {
                        int i_cell = iw_segment[i_seg].indx[0];
                        int j_cell = iw_segment[i_seg].indx[1];
                        int k_cell = iw_segment[i_seg].indx[2];
                        TxVector<double> disp_3d(disp[0], disp[1], 0);
                        TxVector<double> frac_disp = disp_3d * fraction[i_seg];

                        //*****************************************************************************************************************
                        node_field->accumulate_I_CUDA(iw_segment[i_seg], frac_disp, q2dt * remainedWt);
                        //*****************************************************************************************************************
                    }
                    species->add_ptcl_CUDA(ptcl_pos, iw_end, vel, remainedWt);
                }
            }
    }
}
