/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2016 OpenFOAM Foundation
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/
#include "pinballRotatingWallVelocityFvPatchVectorField.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(p, iF),
      origin_a_(),
      origin_b_(),
      origin_c_(),
      axis_(Zero),
      probes_(initializeProbes())
{
}

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const dictionary &dict)
    : fixedValueFvPatchField<vector>(p, iF, dict, false),
      origin_a_(dict.get<vector>("origin_a")),
      origin_b_(dict.get<vector>("origin_b")),
      origin_c_(dict.get<vector>("origin_c")),
      axis_(dict.get<vector>("axis")),
      train_(dict.get<bool>("train")),
      interval_(dict.get<int>("interval")),
      start_time_(dict.get<scalar>("startTime")),
      start_iter_(0),
      policy_name_(dict.get<word>("policy")),
      policy_(torch::jit::load(policy_name_)),
      abs_omega_max_a_(dict.get<scalar>("absOmegaMax_a")),
      abs_omega_max_b_(dict.get<scalar>("absOmegaMax_b")),
      abs_omega_max_c_(dict.get<scalar>("absOmegaMax_c")),
      seed_(dict.get<int>("seed")),
      probes_name_(dict.get<word>("probesDict")),
      gen_(seed_),
      omega_a_(0.0),
      omega_old_a_(0),
      omega_b_(0.0),
      omega_old_b_(0),
      omega_c_(0.0),
      omega_old_c_(0),
      control_time_(0.0),
      update_omega_(false),
      probes_(initializeProbes())
{
   updateCoeffs();
}


Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const pinballRotatingWallVelocityFvPatchVectorField &ptf,
        const fvPatch &p,
        const DimensionedField<vector, volMesh> &iF,
        const fvPatchFieldMapper &mapper)
    : fixedValueFvPatchField<vector>(ptf, p, iF, mapper),
      origin_a_(ptf.origin_a_),
      origin_b_(ptf.origin_b_),
      origin_c_(ptf.origin_c_),
      axis_(ptf.axis_),
      train_(ptf.train_),
      interval_(ptf.interval_),
      start_time_(ptf.start_time_),
      start_iter_(ptf.start_iter_),
      policy_name_(ptf.policy_name_),
      policy_(ptf.policy_),
      abs_omega_max_a_(ptf.abs_omega_max_a_),
      abs_omega_max_b_(ptf.abs_omega_max_b_),
      abs_omega_max_c_(ptf.abs_omega_max_c_),
      seed_(ptf.seed_),
      probes_name_(ptf.probes_name_),
      gen_(ptf.gen_),
      omega_a_(ptf.omega_a_),
      omega_old_a_(ptf.omega_old_a_),
      omega_b_(ptf.omega_b_),
      omega_old_b_(ptf.omega_old_b_),
      omega_c_(ptf.omega_c_),
      omega_old_c_(ptf.omega_old_c_),
      control_time_(ptf.control_time_),
      update_omega_(ptf.update_omega_),
      probes_(initializeProbes())
{
}

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const pinballRotatingWallVelocityFvPatchVectorField &rwvpvf)
    : fixedValueFvPatchField<vector>(rwvpvf),
      origin_a_(rwvpvf.origin_a_),
      origin_b_(rwvpvf.origin_b_),
      origin_c_(rwvpvf.origin_c_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      interval_(rwvpvf.interval_),
      start_time_(rwvpvf.start_time_),
      start_iter_(rwvpvf.start_iter_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_a_(rwvpvf.abs_omega_max_a_),
      abs_omega_max_b_(rwvpvf.abs_omega_max_b_),
      abs_omega_max_c_(rwvpvf.abs_omega_max_c_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      omega_a_(rwvpvf.omega_a_),
      omega_old_a_(rwvpvf.omega_old_a_),
      omega_b_(rwvpvf.omega_b_),
      omega_old_b_(rwvpvf.omega_old_b_),
      omega_c_(rwvpvf.omega_c_),
      omega_old_c_(rwvpvf.omega_old_c_),
      control_time_(rwvpvf.control_time_),
      update_omega_(rwvpvf.update_omega_),
      probes_(initializeProbes())
{
}

Foam::pinballRotatingWallVelocityFvPatchVectorField::
    pinballRotatingWallVelocityFvPatchVectorField(
        const pinballRotatingWallVelocityFvPatchVectorField &rwvpvf,
        const DimensionedField<vector, volMesh> &iF)
    : fixedValueFvPatchField<vector>(rwvpvf, iF),
      origin_a_(rwvpvf.origin_a_),
      origin_b_(rwvpvf.origin_b_),
      origin_c_(rwvpvf.origin_c_),
      axis_(rwvpvf.axis_),
      train_(rwvpvf.train_),
      interval_(rwvpvf.interval_),
      start_time_(rwvpvf.start_time_),
      start_iter_(rwvpvf.start_iter_),
      policy_name_(rwvpvf.policy_name_),
      policy_(rwvpvf.policy_),
      abs_omega_max_a_(rwvpvf.abs_omega_max_a_),
      abs_omega_max_b_(rwvpvf.abs_omega_max_b_),
      abs_omega_max_c_(rwvpvf.abs_omega_max_c_),
      seed_(rwvpvf.seed_),
      probes_name_(rwvpvf.probes_name_),
      gen_(rwvpvf.gen_),
      omega_a_(rwvpvf.omega_a_),
      omega_old_a_(rwvpvf.omega_old_a_),
      omega_b_(rwvpvf.omega_b_),
      omega_old_b_(rwvpvf.omega_old_b_),
      omega_c_(rwvpvf.omega_c_),
      omega_old_c_(rwvpvf.omega_old_c_),
      control_time_(rwvpvf.control_time_),
      update_omega_(rwvpvf.update_omega_),
      probes_(initializeProbes())
{
}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::pinballRotatingWallVelocityFvPatchVectorField::updateCoeffs()
{
    if (updated())
    {
        return;
    }

    const fvMesh& mesh(patch().boundaryMesh().mesh());
    
    

    label patchID = mesh.boundaryMesh().findPatchID("cylinders");

    // Info << "PatchID_mesh " << patchID << endl;

    const polyPatch& cPatch = mesh.boundaryMesh()[patchID];
    const surfaceScalarField& magSf = mesh.magSf();
    const surfaceVectorField& Cf = mesh.Cf();
    const surfaceVectorField& Sf = mesh.Sf();
    
    // Info << "Cf_mesh" << Cf.boundaryField()[patchID] << endl;

    DynamicList <vector> list_centres_a;
    DynamicList <vector> list_centres_b;
    DynamicList <vector> list_centres_c;
    
    DynamicList <vector> list_normals_a;
    DynamicList <vector> list_normals_b;
    DynamicList <vector> list_normals_c;
    
    DynamicList <label> list_labels_a;
    DynamicList <label> list_labels_b;
    DynamicList <label> list_labels_c;
    
    const scalar radius_ = 0.5;
    double distance_a = 0.0;
    double distance_b = 0.0;
    double distance_c = 0.0;
    scalar x = 0.0;
    scalar y= 0.0;
    forAll(cPatch, facei)
    {
    x = Cf.boundaryField()[patchID][facei].x();
    y = Cf.boundaryField()[patchID][facei].y();
    
    distance_a = std::sqrt(pow(x - origin_a_[0], 2) + pow(y - origin_a_[1], 2));
    distance_b = std::sqrt(pow(x - origin_b_[0], 2) + pow(y - origin_b_[1], 2));
    distance_c = std::sqrt(pow(x - origin_c_[0], 2) + pow(y - origin_c_[1], 2));

    // Info << "distance_a " << distance_a << endl;
    // Info << "distance_b " << distance_b << endl;
    // Info << "distance_c " << distance_c << endl;

    if (distance_a < 1.2*radius_)
    {
        list_centres_a.append(Cf.boundaryField()[patchID][facei]);
        list_normals_a.append(Sf.boundaryField()[patchID][facei]/magSf.boundaryField()[patchID][facei]);
        list_labels_a.append(facei);
    }

    if (distance_b < 1.2*radius_)
    {
        list_centres_b.append(Cf.boundaryField()[patchID][facei]);
        list_normals_b.append(Sf.boundaryField()[patchID][facei]/magSf.boundaryField()[patchID][facei]);
        list_labels_b.append(facei);
    }

    if (distance_c < 1.2*radius_)
    {
        list_centres_c.append(Cf.boundaryField()[patchID][facei]);
        list_normals_c.append(Sf.boundaryField()[patchID][facei]/magSf.boundaryField()[patchID][facei]);
        list_labels_c.append(facei);
    }
    }

    // Info << "List labels c: " << list_labels_c << endl;
    // Info << "List labels b: " << list_labels_b << endl;
    // Info << "List labels a: " << list_labels_a << endl;

    // forAll(list_labels_a, facei)

    // {
    //   Info << "facei: " << facei << endl;  
    // }

    // read list of face labels from faceZones for cylinder a,b,c
    // const fvMesh& mesh(patch().boundaryMesh().mesh());

    // const label & faceZoneID_c = mesh.faceZones().findZoneID("faceZone_c");
    // const faceZone& zone_c = mesh.faceZones()[faceZoneID_c];
    // const faceZoneMesh& zoneMesh_c = zone_c.zoneMesh();
    // const labelList& facesZone_c = zoneMesh_c[faceZoneID_c];

    // const label & faceZoneID_b = mesh.faceZones().findZoneID("faceZone_b");
    // const faceZone& zone_b = mesh.faceZones()[faceZoneID_b];
    // const faceZoneMesh& zoneMesh_b = zone_b.zoneMesh();
    // const labelList& facesZone_b = zoneMesh_b[faceZoneID_b];

    // const label & faceZoneID_a = mesh.faceZones().findZoneID("faceZone_a");
    // const faceZone& zone_a = mesh.faceZones()[faceZoneID_a];
    // const faceZoneMesh& zoneMesh_a = zone_a.zoneMesh();
    // const labelList& facesZone_a = zoneMesh_a[faceZoneID_a];

    // // compute face centres (and face normals) 
    // DynamicList <vector> list_centres_c;
    // DynamicList <vector> list_centres_b;
    // DynamicList <vector> list_centres_a;

    // // forAll(facesZone_c, i)
    // forAll(facesZone_c, i)
    // {
    // label faceI = facesZone_c[i];
    // label facePatchID = mesh.boundaryMesh().whichPatch(faceI);
    // label faceID = mesh.boundaryMesh()[facePatchID].whichFace(facesZone_c[faceI]);  
    // const Vector <double> Cf_c = mesh.Cf().boundaryField()[facePatchID][faceID];
    // list_centres_c.append(Cf_c);
    // }

    // // forAll(facesZone_b, i)
    // forAll(facesZone_b, i)
    // {
    // label faceI = facesZone_b[i];
    // label facePatchID = mesh.boundaryMesh().whichPatch(faceI);
    // label faceID = mesh.boundaryMesh()[facePatchID].whichFace(facesZone_b[faceI]); 
    // const Vector <double> Cf_b = mesh.Cf().boundaryField()[facePatchID][faceID];
    // list_centres_b.append(Cf_b);
    // }

    // //forAll(facesZone_a, i)
    // forAll(facesZone_a, i)
    // {
    // label faceI = facesZone_a[i];
    // label facePatchID = mesh.boundaryMesh().whichPatch(faceI);
    // label faceID = mesh.boundaryMesh()[facePatchID].whichFace(facesZone_a[faceI]); 
    // const Vector <double> Cf_a = mesh.Cf().boundaryField()[facePatchID][faceID];
    // list_centres_a.append(Cf_a);
    // }

    // update angular velocity
    const scalar t = this->db().time().timeOutputValue();
    
    Info << "Works until here: " << endl;

    bool steps_remaining = (this->db().time().timeIndex() - start_iter_) % interval_ == 0;
    if (t >= start_time_)
    {
        if(start_iter_ == 0)
        {
            start_iter_ = this->db().time().timeIndex();
            steps_remaining = true;
        }

        if (steps_remaining && update_omega_)
        {
            omega_old_a_ = omega_a_;
            omega_old_b_ = omega_b_;
            omega_old_c_ = omega_c_;
            control_time_ = t;

            Info << "Works until here: " << endl;

            const volScalarField& p = this->db().lookupObject<volScalarField>("p"); 
            scalarField p_sample = probes_.sample(p);

            if (Pstream::master()) // evaluate policy only on the master
            {

                torch::Tensor features = torch::from_blob(
                    p_sample.data(), {1, p_sample.size()}, torch::TensorOptions().dtype(torch::kFloat64)
                );
                std::vector<torch::jit::IValue> policyFeatures{features};
                torch::Tensor dist_parameters = policy_.forward(policyFeatures).toTensor();
                scalar alpha_a = dist_parameters[0][0].item<double>();
                scalar alpha_b = dist_parameters[0][1].item<double>();
                scalar alpha_c = dist_parameters[0][2].item<double>();
                scalar beta_a = dist_parameters[0][3].item<double>();
                scalar beta_b = dist_parameters[0][4].item<double>();
                scalar beta_c = dist_parameters[0][5].item<double>();

                Info << "Read parameters: " << endl;

                Info << "alpha_a: " << alpha_a << endl;
                Info << "alpha_b: " << alpha_b << endl;
                Info << "alpha_c: " << alpha_c << endl;
                Info << "beta_a: " << beta_a << endl;
                Info << "beta_b: " << beta_b << endl;
                Info << "beta_c: " << beta_c << endl;

                std::gamma_distribution<double> distribution_1_a(alpha_a, 1.0);
                std::gamma_distribution<double> distribution_2_a(alpha_b, 1.0);
                std::gamma_distribution<double> distribution_3_a(alpha_c, 1.0);
                std::gamma_distribution<double> distribution_1_b(beta_a, 1.0);
                std::gamma_distribution<double> distribution_2_b(beta_b, 1.0);
                std::gamma_distribution<double> distribution_3_b(beta_c, 1.0);
                scalar omega_pre_scale_a;
                scalar omega_pre_scale_b;
                scalar omega_pre_scale_c;
                if (train_)
                {
                    // sample from Beta distribution during training
                    double number_1_a = distribution_1_a(gen_);
                    double number_2_a = distribution_2_a(gen_);
                    double number_3_a = distribution_3_a(gen_);
                    double number_1_b = distribution_1_b(gen_);
                    double number_2_b = distribution_2_b(gen_);
                    double number_3_b = distribution_3_b(gen_);

                    Info << "number_1_a: " << number_1_a << endl;
                    Info << "number_2_a: " << number_2_a << endl;
                    Info << "number_3_a: " << number_3_a << endl;
                    Info << "number_1_b: " << number_1_b << endl;
                    Info << "number_2_b: " << number_2_b << endl;
                    Info << "number_3_b: " << number_3_b << endl;

                    omega_pre_scale_a = number_1_a / (number_1_a + number_1_b);
                    omega_pre_scale_b = number_2_a / (number_2_a + number_2_b);
                    omega_pre_scale_c = number_3_a / (number_3_a + number_3_b);
                }
                else
                {
                    // use expected (mean) angular velocity
                    omega_pre_scale_a = alpha_a / (alpha_a + beta_a);
                    omega_pre_scale_b = alpha_b / (alpha_b + beta_b);
                    omega_pre_scale_c = alpha_c / (alpha_c + beta_c);
                }
                // rescale to actionspace
                omega_a_ = (omega_pre_scale_a - 0.5) * 2 * abs_omega_max_a_;
                omega_b_ = (omega_pre_scale_b - 0.5) * 2 * abs_omega_max_b_;
                omega_c_ = (omega_pre_scale_c - 0.5) * 2 * abs_omega_max_c_;
                // save trajectory
                saveTrajectory(alpha_a, beta_a, alpha_b, beta_b, alpha_c, beta_c);
                Info << "New omega_a: " << omega_a_ << "; old value: " << omega_old_a_ << "\n";
                Info << "New omega_b: " << omega_b_ << "; old value: " << omega_old_b_ << "\n";
                Info << "New omega_c: " << omega_c_ << "; old value: " << omega_old_c_ << "\n";
            }
            Pstream::scatter(omega_a_);
            Pstream::scatter(omega_b_);
            Pstream::scatter(omega_c_);
            //, omega_b_, omega_c_);

            // avoid update of angular velocity during p-U coupling
            update_omega_ = false;
        }
    }

    // activate update of angular velocity after p-U coupling
    if (!steps_remaining)
    {
        update_omega_ = true;
    }

    // update angular velocity by linear transition from old to new value
    const scalar dt = this->db().time().deltaTValue();
    scalar d_omega_a = (omega_a_ - omega_old_a_) / (dt * interval_) * (t - control_time_);
    scalar omega_a = omega_old_a_ + d_omega_a;
    scalar d_omega_b = (omega_b_ - omega_old_b_) / (dt * interval_) * (t - control_time_);
    scalar omega_b = omega_old_b_ + d_omega_b;
    scalar d_omega_c = (omega_c_ - omega_old_c_) / (dt * interval_) * (t - control_time_);
    
    Info << "d_omega_c: " << d_omega_c << endl;
    
    scalar omega_c = omega_old_c_ + d_omega_c;
    
    // Calculate the rotating wall velocity from the specification of the motion
    // const vectorField Up(sizeof(facesZone_a)+sizeof(facesZone_b)+sizeof(facesZone_c)) = 0;  
    
    Info << "omega_c: " << omega_c << endl;
    Info << "omega_b: " << omega_b << endl;
    Info << "omega_a: " << omega_a << endl;
    
    // vectorField Up(374);
    // Info << "Up allocated: " << Up << endl;
    // for (int i=0; i<100; i++) 
    // {   label facei = list_labels_a[i];
    //     Up[facei] = ((-omega_a) * (list_centres_a[i] - origin_a_) ^ (axis_ / mag(axis_)));
    //     // Info << "Up tangential: " << Up[facei] << endl;
    //     //Up[facei] = (Up[facei] - list_normals_a[i] * (list_normals_a[i] & Up[facei]));
    //     // Info << "Works until here: " << endl;
    //     // Up[facei] = list_centres_a[i];
    //     // Info << "Up cleaned: " << Up[facei] << endl;
    // }
    // for (int i=100; i<237; i++) 
    // {   label facei = list_labels_b[i-100];
    //     // const vector Up_b
    //     Up[facei] = ((-omega_b) * (list_centres_b[i-100] - origin_b_) ^ (axis_ / mag(axis_)));
    //     //Up[facei] = (Up[facei] - list_normals_b[i-100] * (list_normals_b[i-100] & Up[facei]));
    //     // Up[facei] = list_centres_b[i];
    // }
    // for (int i=237; i<374; i++) 
    // {   label facei = list_labels_c[i-237];
    //     Up[facei] = ((-omega_c) * (list_centres_c[i-237] - origin_c_) ^ (axis_ / mag(axis_)));
    //     //Up[facei] = (Up[facei] - list_normals_c[i-237] * (list_normals_c[i-237] & Up[facei]));
    //     // Up[facei] = list_centres_c[i];
    // }

    const int patch_size = (patch().Cf()).size();
    
    Info << "Patch size: " << patch_size << endl;

    vectorField Up(patch_size);
    // Info << "Up allocated: " << Up << endl;
    // for (int i=0; i<100; i++) 
    forAll(list_labels_a, index)
    {   label facei = list_labels_a[index];
        //Up[facei] = (-omega_a) * ((Cf.boundaryField()[patchID][facei] - origin_a_) ^ (axis_ / mag(axis_)));
        // Info << "Face center a: " << list_centres_a[index] << endl;
        Up[facei] = (-omega_a) * ((list_centres_a[index] - origin_a_) ^ (axis_ / mag(axis_)));
        //Info << "Up tangential: " << Up[facei] << endl;
        // Up[facei] = (Up[facei] - list_normals_a[index] * (list_normals_a[index] & Up[facei]));
        // Info << "Works until here: " << endl;
        // Up[facei] = list_centres_a[i];
        // Info << "Up cleaned: " << Up[facei] << endl;
    }
    // for (int i=100; i<237; i++) 
    forAll(list_labels_b, index)
    {   //label facei = list_labels_b[i-100];
        label facei = list_labels_b[index];
        //Up[facei] = ((-omega_b) * (list_centres_b[i-100] - origin_b_) ^ (axis_ / mag(axis_)));
        //Up[facei] = (-omega_b) * ((Cf.boundaryField()[patchID][facei] - origin_b_) ^ (axis_ / mag(axis_)));
        // Info << "Face center b: " << list_centres_b[index] << endl;
        Up[facei] = (-omega_b) * ((list_centres_b[index] - origin_b_) ^ (axis_ / mag(axis_)));
        //Info << "Up tangential: " << Up[facei] << endl;
        // Up[facei] = (Up[facei] - list_normals_b[index] * (list_normals_b[index] & Up[facei]));
        // Up[facei] = list_centres_b[i];
    }
    //for (int i=237; i<374; i++) 
    forAll(list_labels_c, index)
    {   //label facei = list_labels_c[i-237];
        label facei = list_labels_c[index];
        //Up[facei] = ((-omega_c) * (list_centres_c[i-237] - origin_c_) ^ (axis_ / mag(axis_)));
        // Up[facei] = (-omega_c) * ((Cf.boundaryField()[patchID][facei] - origin_c_) ^ (axis_ / mag(axis_)));
        // Info << "Face center c: " << list_centres_c[index] << endl;
        Up[facei] = (-omega_c) * ((list_centres_c[index] - origin_c_) ^ (axis_ / mag(axis_)));
        //Info << "Up tangential: " << Up[facei] << endl;
        // Up[facei] = (Up[facei] - list_normals_c[index] * (list_normals_c[index] & Up[facei]));
        // Up[facei] = list_centres_c[i];
    }

    // Info << "Up filled: " << Up << endl;

    vectorField::operator=(Up);

    // const int a = (patch().Cf()).size();

    // Info << "a: " << a << endl;

    // const vectorField Up(
    //     (-omega_a) * ((patch().Cf() - origin_a_) ^ (axis_ / mag(axis_))));

    // // const vectorField n(patch().nf());
    // // vectorField::operator=(Up - n * (n & Up));

    // vectorField::operator=(Up);

    // forAll(Up, i)
    // {
    // const vectorField Up_c(
    //     (-omega_c) * (list_centres_c - origin_c_) ^ (axis_ / mag(axis_)));

    // const vectorField Up_b(
    //     (-omega_b) * (list_centres_b - origin_b_) ^ (axis_ / mag(axis_)));

    // const vectorField Up_a(
    //     (-omega_a) * (list_centres_a - origin_a_) ^ (axis_ / mag(axis_)));
    
    
    
    // Remove the component of Up normal to the wall
    // just in case it is not exactly circular
    // const vectorField n(patch().nf());
    // vectorField::operator=(Up - n * (n & Up));

    fixedValueFvPatchVectorField::updateCoeffs();

    
}

void Foam::pinballRotatingWallVelocityFvPatchVectorField::write(Ostream &os) const
{
    fvPatchVectorField::write(os);
    os.writeEntry("origin_a", origin_a_);
    os.writeEntry("origin_b", origin_b_);
    os.writeEntry("origin_c", origin_c_);
    os.writeEntry("axis", axis_);
    os.writeEntry("policy", policy_name_);
    os.writeEntry("startTime", start_time_);
    os.writeEntry("interval", interval_);
    os.writeEntry("train", train_);
    os.writeEntry("absOmegaMax_a", abs_omega_max_a_);
    os.writeEntry("absOmegaMax_b", abs_omega_max_b_);
    os.writeEntry("absOmegaMax_c", abs_omega_max_c_);
    os.writeEntry("seed", seed_);
    os.writeEntry("probesDict", probes_name_);
}

void Foam::pinballRotatingWallVelocityFvPatchVectorField::saveTrajectory(scalar alpha_a, scalar beta_a, scalar alpha_b, scalar beta_b, scalar alpha_c, scalar beta_c) const
{
    std::ifstream file("trajectory.csv");
    std::fstream trajectory("trajectory.csv", std::ios::app | std::ios::binary);
    const scalar t = this->db().time().timeOutputValue();
    scalar dt = 0.0;
    if (start_iter_ == 1)
    {
        // subtract dt to be consistent with the function object output when controlling from the beginning
        dt = this->db().time().deltaTValue();
    }

    if(!file.good())
    {
        // write header
        trajectory << "t, omega_a, alpha_a, beta_a, omega_b, alpha_b, beta_b, omega_c, alpha_c, beta_c";
    }

    trajectory << std::setprecision(15)
               << "\n"
               << t - dt << ", "
               << omega_a_ << ", "
               << alpha_a << ", "
               << beta_a << ", "
               << omega_b_ << ", "
               << alpha_b << ", "
               << beta_b << ", "
               << omega_c_ << ", "
               << alpha_c << ", "
               << beta_c;
               

}

Foam::probes Foam::pinballRotatingWallVelocityFvPatchVectorField::initializeProbes()
{
    const dictionary& funcDict = this->db().time().controlDict().subDict("functions");
    if (!funcDict.found(probes_name_))
    {
        FatalError << "probesDict" << probes_name_ << " not found\n" << exit(FatalError);
        
    }
    const dictionary& probesDict = funcDict.subDict(probes_name_);
    return Foam::probes("probes", this->db().time(), probesDict, false, true);
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
    makePatchTypeField(
        fvPatchVectorField,
        pinballRotatingWallVelocityFvPatchVectorField);
}

// ************************************************************************* //
