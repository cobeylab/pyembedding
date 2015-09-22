/*
 *   This file is part of TISEAN
 *
 *   Copyright (c) 1998-2007 Rainer Hegger, Holger Kantz, Thomas Schreiber
 *
 *   TISEAN is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation; either version 2 of the License, or
 *   (at your option) any later version.
 *
 *   TISEAN is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with TISEAN; if not, write to the Free Software
 *   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */
/* Author: Lucas C. Uzal. Last modified: Jan 11, 2011 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include "routines/tsa.h"

#define WID_STR "Computes a cost function for evaluating attractor reconstructions"

char *outfile=NULL;
char *infile=NULL;
char *infile2=NULL;
unsigned int column=1;
unsigned long length=ULONG_MAX,length0=ULONG_MAX;
int exclude=0,theiler=-1;
int mdim=2,Mdim=7,dim;
unsigned int verbosity=0xff;
unsigned int method=1;
double EPSM=0.01;
double *series;
double *x;
double stdev_data,variance_data;
double **barx;
double **base;

#define BOX 1024
unsigned long ibox=BOX-1;
unsigned long **box,*list;
int **boxsize; 
int xdim=0,ydim=1;
unsigned long **nearest;
double *dnearest;
double *eps2;
char local=0;
int *delaylist;
int Mtw=-1;
int mtw=1;
int lmax;
int Tm=-1;
int hdiv=500;
double temp;
char *outfile_amp=NULL;
char *outfile_min=NULL;
unsigned long *scr;
int Nref=-1;
int NN=2;
double dimstep=0.15,twstep=5.0;
char dimset=0;

FILE *file=NULL;

double compute_sigma2(void);
void show_options(char *progname);
void scan_options(int n,char **in);
void shuffle(unsigned long *v, int n);
double distance(int n, int m);
void rescale(double **barx,int dim,unsigned long l,int *j1, int *j2);
int automatic_Tm(double *x,unsigned long l,double std);
void mmb(double eps);
int order(int element, double dx, unsigned long *knn, double *dknn, int m, int M);
int find_nearest(int n, unsigned long *knn, double *dknn, double eps, int M);
double E2_x(double *x, unsigned long *nlist, int kmax, int T0, int Tm);
double epsilon2(unsigned long *nlist, int kmax);
void fill_delay_list(int d, int m);
void increase(int *j, double delta);
void Legendre_basis(int p, int q);
void embed1(double *x, int m);
void embed2(double *x, int tw, int m);

int main(int argc,char **argv)
{
	FILE *file_min=NULL;
	double min,inter=0.0,av;
	int i,k,m,idim;
	int p,mp,d,md,t,tw;
	char ext[30];
	double sigma2,Lk,Lkmin;
	double *minL;
	int *mintw,*dimarray,ldim;
	int dimmin,twmin;
	

	if (scan_help(argc,argv)) show_options(argv[0]);
	scan_options(argc,argv);
	#ifndef OMIT_WHAT_I_DO
	if (verbosity&VER_INPUT) what_i_do(argv[0],WID_STR);
	#endif

	infile=search_datafile(argc,argv,NULL,verbosity);

	if(outfile == NULL){
		if(infile == NULL){
				check_alloc(outfile=(char*)calloc((size_t)30,(size_t)1));
				sprintf(outfile,"stdin");
		}else{
				check_alloc(outfile=(char*)calloc(strlen(infile)+20,(size_t)1));
				strcpy(outfile,infile);
		}
	}
	check_alloc(outfile_amp=(char*)calloc(strlen(outfile)+20,(size_t)1));
	check_alloc(outfile_min=(char*)calloc(strlen(outfile)+20,(size_t)1));
		
	if(local==1) {Mtw=mtw; Mdim=mdim;}
	if(method==0){
		if(infile2 == NULL){fprintf(stderr,"NO INPUT FILE OPTION -i\n");exit(0);}
		argv[1]=infile2; // trick!
		infile2=search_datafile(2,argv,NULL,verbosity);
		barx=(double**)get_multi_series(infile2,&length0,0,&Mdim,"",dimset,verbosity);
		Mtw=0;
	}
	series=(double*)get_series(infile,&length,exclude,column,verbosity);
	if((length!=length0)&&(method==0)){
		fprintf(stderr,"The length of the series does not match"
		" with the length of the reconstructed orbit file. "
		"Use -x and -l options to solve the problem.\n");exit(1);
	}
		
	rescale_data(series,length,&min,&inter);
	variance(series,length,&av,&stdev_data);
	variance_data=stdev_data*stdev_data;
	fprintf(stderr,"Standar Deviation: %lf\n Inter: %lf\n",stdev_data*inter,inter);
	if(Tm<0) Tm=automatic_Tm(series,length,stdev_data);
	if(theiler<0) theiler=Tm/2+Tm%2;
	if(Mtw<0) if((Mtw=Tm*10)>length/10) Mtw=length/5;
	lmax=length-Mtw-Tm-1;
	if (lmax<1){fprintf(stderr,"NOT ENOUGH POINTS\n");exit(FALSE_NEAREST_NOT_ENOUGH_POINTS);}
	if(local==1) Nref=lmax;
	else{
		if((Nref<1)||(Nref>lmax)) if((Nref=length/2)>5000) Nref=5000;
		if(Nref>lmax) Nref=lmax;
	}
	x=series+Mtw;
	
	check_alloc(box=(unsigned long**)malloc(sizeof(long*)*BOX));
	for (i=0;i<BOX;i++) check_alloc(box[i]=(unsigned long*)malloc(sizeof(long)*BOX));
	check_alloc(boxsize=(int**)malloc(sizeof(int*)*BOX));
	for (i=0;i<BOX;i++) check_alloc(boxsize[i]=(int*)malloc(sizeof(int)*BOX));
	check_alloc(nearest=(unsigned long**)malloc(sizeof(long*)*(Nref)));
	for(i=0;i<Nref;i++) check_alloc(nearest[i]=(unsigned long*)malloc(sizeof(long)*(NN+1)));
	check_alloc(dnearest=(double*)malloc(sizeof(double)*(NN)));
	check_alloc(eps2=(double*)malloc(sizeof(double)*(Nref)));
	check_alloc(scr=(unsigned long*)malloc(sizeof(long)*(length)));
	check_alloc(list=(unsigned long*)malloc(sizeof(long)*(length)));
	
	if(local==0){

		if(method!=2){
			for(ldim=0,dim=mdim;dim<=Mdim;increase(&dim,dimstep),ldim++);
			check_alloc(mintw=(int*)malloc(sizeof(int)*(ldim)));
			check_alloc(minL=(double*)malloc(sizeof(double)*(ldim)));
			check_alloc(dimarray=(int*)malloc(sizeof(int)*(ldim)));
		}
		else Mdim=Mtw+1;
				
		strcpy(outfile_amp,outfile);		sprintf(ext,".amp");
		strcat(outfile_amp,ext);		test_outfile(outfile_amp);
		strcpy(outfile_min,outfile);		sprintf(ext,".min");
		strcat(outfile_min,ext);		test_outfile(outfile_min);
		// Open files
		file=fopen(outfile_amp,"w");
		file_min=fopen(outfile_min,"w");
		if (verbosity&VER_INPUT){
			fprintf(stderr,"Opened %s for writing\n",outfile_amp);
			fprintf(stderr,"Opened %s for writing\n",outfile_min);
			fprintf(stderr,"Using %d reference points\n",Nref);
			fprintf(stderr,"Using T_M=%d\n",Tm);
			fprintf(stderr,"Using ThW=%d\n",theiler);
			fprintf(stderr,"Using k=%d neighbours\n",NN);
			if(method>0) fprintf(stderr,"%d<=tw<=%d\n",mtw,Mtw);
			if(method!=2) fprintf(stderr,"%d<=m<=%d\n",mdim,Mdim);
		}

		// cargo el vector de mezcla
		for (i=0;i<lmax;i++) scr[i]=i;
		srand(1); // Inicio semilla
		if(method>0){
			check_alloc(barx=(double**)malloc(sizeof(double*)*(Mdim)));
			if(method<3){
				check_alloc(delaylist=(int*)malloc(sizeof(int)*(Mdim)));
				if(method==1) fprintf(file,"# delay");
			}
			else{
				check_alloc(base=(double**)malloc(sizeof(double*)*(Mdim)));
				for(dim=0;dim<Mdim;dim++){
					check_alloc(base[dim]=(double*)malloc(sizeof(double)*(Mtw)));
					check_alloc(barx[dim]=(double*)malloc(sizeof(double)*(lmax)));
				}
				fprintf(file,"# t_w");
			}
		}

		// Inicialización de variables
		Lkmin=1e60;
		twstep/=100;
		fprintf(stderr,"m=");
		if(method==2){
			fprintf(file,"# tw L\n");
			fprintf(stderr,"tw+1\n");
		}
		else {
			for(idim=0,dim=mdim;idim<ldim;increase(&dim,dimstep),idim++) {
				dimarray[idim]=dim;
				minL[idim]=1e60;
				fprintf(file," L(m=%d)",dim);
				fprintf(stderr,"%d,",dim);
			}
			fprintf(file,"\n");
			fprintf(stderr,"\n");
		}
		if(method==0){
			for(idim=0;idim<ldim;idim++){
				shuffle(scr,lmax);// mezclo el vector scr
				dim=dimarray[idim];
				rescale(barx,dim,length0,&xdim,&ydim);
				sigma2=compute_sigma2();
				Lk=log10(sqrt(sigma2));
				if(Lk<Lkmin){ Lkmin=Lk; twmin=tw; dimmin=dim;}
				if(Lk<minL[idim]){ minL[idim]=Lk; mintw[idim]=tw;}
			}
		}
		if(method==1){
			fprintf(stderr,"d=");
			if((md=mtw/(Mdim-1))<1) md=1;
			for(d=md,tw=((mdim-1)*d);tw<=Mtw;increase(&d,twstep),tw=((mdim-1)*d)){
				fprintf(stderr,"%d,",d);  fflush(stderr);
				fill_delay_list(d,Mdim); // llenar la lista de delays
				embed1(x,Mdim);
				fprintf(file,"%d",d);
				shuffle(scr,lmax);// mezclo el vector scr
				for(idim=0;idim<ldim;idim++){
					dim=dimarray[idim];
					ydim=dim-1; tw=ydim*d;// para inicializar cajas
					if(tw>Mtw) break;
					sigma2=compute_sigma2();
					Lk=log10(sqrt(sigma2));
					if(Lk<Lkmin){ Lkmin=Lk; twmin=tw; dimmin=dim;}
					if(Lk<minL[idim]){ minL[idim]=Lk; mintw[idim]=tw;}
					fprintf(file," %e",Lk);
				}
				fprintf(file,"\n");  fflush(file);
			}
			fprintf(stderr,"\n");  fflush(stderr);
		}
		if(method==2){
			fprintf(stderr,"tw=");
			fill_delay_list(d=1,Mdim); // llenar la lista de delays
			embed1(x,Mdim);
			for(tw=mtw;tw<=Mtw;increase(&tw,twstep)){
				fprintf(stderr,"%d,",tw);  fflush(stderr);
				dim=tw+1;
				fprintf(file,"%d",tw);
				shuffle(scr,lmax);// mezclo el vector scr
				sigma2=compute_sigma2();
				Lk=log10(sqrt(sigma2));
				if(Lk<Lkmin){ Lkmin=Lk; twmin=tw; dimmin=dim;}
				fprintf(file," %e\n",Lk); fflush(file);
			}
			fprintf(stderr,"\n");  fflush(stderr);
		}
		if(method==3){
			fprintf(stderr,"tw=");
			if((mp=mtw/2)<1) mp=0;
			for(p=mp,tw=2*p+1;tw<=Mtw;increase(&p,twstep),tw=2*p+1){
				fprintf(stderr,"%d,",tw);  fflush(stderr);
				Legendre_basis(p,Mdim);
				embed2(x,tw+1,Mdim);
				fprintf(file,"%d",tw);
				shuffle(scr,lmax);// mezclo el vector scr
				for(idim=0;idim<ldim;idim++)
				{
					dim=dimarray[idim];
					if(dim>tw+1) break;
					sigma2=compute_sigma2();
					Lk=log10(sqrt(sigma2));
					if(Lk<Lkmin){ Lkmin=Lk; twmin=tw; dimmin=dim;}
					if(Lk<minL[idim]){ minL[idim]=Lk; mintw[idim]=tw;}
					fprintf(file," %e",Lk);
				}
				fprintf(file,"\n");  fflush(file);
			}
			fprintf(stderr,"\n");  fflush(stderr);
		}
		fclose(file);

		if(method!=2){
			fprintf(file_min,"#dim Lmin twmin\n");
			for(idim=0;idim<ldim;idim++)
				fprintf(file_min,"%d %e %d\n",dim=dimarray[idim],minL[idim],mintw[idim]);
			fclose(file_min);
		}
		
		fprintf(stderr,"\nMin: %e at tw=%d; m=%d\n\n",Lkmin,twmin,dimmin);
	}
	else{
		Mdim=mdim; Mtw=mtw;
		strcpy(outfile_amp,outfile);		sprintf(ext,".loc");
		strcat(outfile_amp,ext);		test_outfile(outfile_amp);
		// Open files
		file=fopen(outfile_amp,"w");
		if (verbosity&VER_INPUT){
			fprintf(stderr,"Computing local cost function\n",Nref);
			fprintf(stderr,"Opened %s for writing\n",outfile_amp);
			fprintf(stderr,"Using T_M=%d\n",Tm);
			fprintf(stderr,"Using ThW=%d\n",theiler);
			fprintf(stderr,"Using k=%d neighbours\n",NN);
			if(method>0) fprintf(stderr,"tw=%d\n",mtw);
			if(method!=2) fprintf(stderr,"m=%d\n",mdim);
		}

		// cargo el vector de mezcla
		for (i=0;i<lmax;i++) scr[i]=i;
		
		if(method>0){
			check_alloc(barx=(double**)malloc(sizeof(double*)*(Mdim)));
			if(method<3) check_alloc(delaylist=(int*)malloc(sizeof(int)*(Mdim)));
			else{
				check_alloc(base=(double**)malloc(sizeof(double*)*(Mdim)));
				for(dim=0;dim<Mdim;dim++){
					check_alloc(base[dim]=(double*)malloc(sizeof(double)*(Mtw)));
					check_alloc(barx[dim]=(double*)malloc(sizeof(double)*(lmax)));
				}
			}
		}

		dim=mdim;
		if(method==0){
			rescale(barx,dim,length0,&xdim,&ydim);
			sigma2=compute_sigma2();
		}
		tw=mtw;
		if(method==1){
			d=tw/(dim-1);
			fill_delay_list(d,dim); // llenar la lista de delays
			embed1(x,dim);
			ydim=dim-1; tw=ydim*d;// para inicializar cajas
			sigma2=compute_sigma2();
		}
		if(method==3){
			p=tw/2;
			Legendre_basis(p,Mdim);
			tw=2*p;
			embed2(x,tw+1,Mdim);
			sigma2=compute_sigma2();
		}
		fclose(file);

		fprintf(stderr,"\nGlobal cost function value: %e\n\n",log10(sqrt(sigma2)));
	}
	
	if (infile != NULL)	 	free(infile);
	if (outfile != NULL)		free(outfile);
	if (outfile_amp != NULL) 	free(outfile_amp);
	if (outfile_min != NULL) 	free(outfile_min);

	if(method==1) free(delaylist);
	if(method==2) free(delaylist);
	if(method==3){
		for(dim=0;dim<Mdim;dim++){free(base[dim]); free(barx[dim]); }
		free(base);
	}
	if(method==0){
		for(dim=0;dim<Mdim;dim++) free(barx[dim]);
	}
	
	free(barx);
	free(series);
	free(scr);
	free(list);
	for (i=0;i<Nref;i++) free(nearest[i]);
	free(nearest);
	free(dnearest);
	for (i=0;i<BOX;i++) free(box[i]);
	free(box);
	for (i=0;i<BOX;i++) free(boxsize[i]);
	free(boxsize);
	if((method!=2)&&(local==0)){
		free(minL);
		free(mintw);
		free(dimarray);
	}
	return 0;
}

double compute_sigma2(void)
{
	int i,j;
	int nref,count;
	double sigma2;
	double snorm,weight;
	double E2,locals2;
	int mfound;

	
	mmb(EPSM);// Inicializa cajas

	snorm=0.0;
	count=0;
	for (j=0;j<Nref;j++){
		i=scr[j];
		nearest[j][0]=i;
		mfound=find_nearest(i,nearest[j]+1,dnearest,EPSM,NN);
		if(mfound!=NN){
			fprintf(stderr,"No enough points found!\n");
			return -1.0;
		}
		if((eps2[j]=epsilon2(nearest[j],NN+1))>0.0){
			snorm+=1./eps2[j];
			count++;
		}
	}
	snorm/=count;
	sigma2=0.0;
	mfound=NN+1;// punto + vecinos
		
	if(local==1) fprintf(file,"# nSigma2 eps2 x1 x2 etc \n");
	
	for(j=0;j<Nref;j++){
		if(eps2[j]>0.0){
			E2=E2_x(x,nearest[j],mfound,1,Tm);
			weight=1./eps2[j];
			sigma2+=(locals2=weight*E2/snorm);
			if(local==1){
				fprintf(file,"%e %e",locals2,eps2[j]);
				for(i=0;i<dim;i++) fprintf(file," %e",barx[i][scr[j]]);
				fprintf(file,"\n");
			}
		}
	}
	return sigma2/count;

}
void show_options(char *progname)
{
	what_i_do(progname,WID_STR);
	fprintf(stderr," Usage: %s [options]\n",progname);
	fprintf(stderr," Options:\n");
	fprintf(stderr,"Everything not being a valid option will be interpreted"
		" as a possible"
		" datafile.\nIf no datafile is given stdin is read. Just - also"
		" means stdin\n");
	fprintf(stderr,"\t-l # of data [default: whole file]\n");
	fprintf(stderr,"\t-x # of lines to ignore [default: 0]\n");
	fprintf(stderr,"\t-c column to read [default: 1]\n");
	fprintf(stderr,"\t-s prediction horizon T_M [default: automatic]\n");
	fprintf(stderr,"\t-k # of neighbours [default: %d]\n",NN);
	fprintf(stderr,"\t-N # of reference points [default: min(5000,(# of data)/2)]\n");
	fprintf(stderr,"\t-t theiler window [default: T_M/2]\n");
	fprintf(stderr,"\t-m min. embedding dimension [default: %d]\n",mdim);  
	fprintf(stderr,"\t-M max. embedding dimension [default: %d]\n",Mdim);
	fprintf(stderr,"\t-w min. time window [default: %d]\n",mtw);
	fprintf(stderr,"\t-W max. time window [default: automatic]\n");
	fprintf(stderr,"\t-%% time window resolution [default: %1.0lf%%]\n",twstep);
	fprintf(stderr,"\t-e reconstruction method [default: 1]\n\t\t"
		"0='read reconstructed orbits from file'\n\t\t"
		"1='delay vector'\n\t\t"
		"2='full delay vector'\n\t\t"
		"3='Legendre coordinates'\n");
	fprintf(stderr,"\t-L compute local cost function [default: do not compute]\n");
	fprintf(stderr,"\t-o output file [default: 'datafile'.amp; without -o"
		" stdout]\n");
	fprintf(stderr,"\t-i imput file [for -e2 option]\n");
	fprintf(stderr,"\t-V verbosity level [default: 3]\n\t\t"
		"0='only panic messages'\n\t\t"
		"1='+ input/output messages'\n\t\t"
		"2='+ information about the current state\n");
	fprintf(stderr,"\t-h show these options\n");
	exit(0);
}

void scan_options(int n,char **in)
{
	char *out;

	if ((out=check_option(in,n,'l','u')) != NULL)
		sscanf(out,"%lu",&length);
	if ((out=check_option(in,n,'x','d')) != NULL)
		sscanf(out,"%d",&exclude);
	if ((out=check_option(in,n,'c','u')) != NULL)
		sscanf(out,"%u",&column);
	if ((out=check_option(in,n,'m','d')) != NULL)
		sscanf(out,"%d",&mdim);
	if ((out=check_option(in,n,'M','d')) != NULL){
		sscanf(out,"%d",&Mdim);
		dimset=1;
	}
	if ((out=check_option(in,n,'s','d')) != NULL)
		sscanf(out,"%d",&Tm);
	if ((out=check_option(in,n,'t','d')) != NULL)
		sscanf(out,"%d",&theiler);
	if ((out=check_option(in,n,'%','f')) != NULL)
		sscanf(out,"%lf",&twstep);
	if ((out=check_option(in,n,'w','d')) != NULL)
		sscanf(out,"%d",&mtw);
	if ((out=check_option(in,n,'W','d')) != NULL)
		sscanf(out,"%d",&Mtw);
	if ((out=check_option(in,n,'N','d')) != NULL)
		sscanf(out,"%d",&Nref);
	if ((out=check_option(in,n,'k','d')) != NULL)
		sscanf(out,"%d",&NN);
	if ((out=check_option(in,n,'e','d')) != NULL)
		sscanf(out,"%d",&method);
	if ((out=check_option(in,n,'L','n')) != NULL)
		local=1;
	if ((out=check_option(in,n,'V','d')) != NULL)
		sscanf(out,"%d",&verbosity);
	if ((out=check_option(in,n,'o','o')) != NULL) {
		if (strlen(out) > 0)
			outfile=out;
	}
	if ((out=check_option(in,n,'i','o')) != NULL) {
		if (strlen(out) > 0)
			infile2=out;
	}
}

#define HISTO_SIZE 10000
#define HISTO_FRACTION 0.05
int automatic_Tm(double *x,unsigned long l,double std)
{
	int i,imax,ib,n,nmax,Tm,j,fraction,cross=0;
	double d,dmax;
	int count1,count2;
	int hist[HISTO_SIZE];
	imax=l/3;
	ib=l;
	dmax=0.25;
	count1=0;
	for(i=1;(i<imax)&&(count1<ib);i++){
		for(j=0;j<HISTO_SIZE;j++) hist[j]=0;
		nmax=l-i;
		fraction=HISTO_FRACTION*nmax;
		for(n=0;n<nmax;n++){
			j=(int)(fabs(x[n]-x[n+i])*HISTO_SIZE);
			if(j>=HISTO_SIZE) j=HISTO_SIZE-1;
			hist[j]++;
		}
		for(j=HISTO_SIZE-1,count2=0;count2<fraction;j--) count2+=hist[j];
		d=((double)j+0.5)/HISTO_SIZE;
		if(d>dmax){
			dmax=d;
			Tm=i;
			if(cross==0){//first cross d2max
				imax=9*i; 
				ib=i/2+1;
				cross=1;
			}
			count1=0;
		}
		else count1++;
	}
	if (count1<ib) Tm=2*ib; // No first maximum founded
	fprintf(stderr,"Automatic selection of T_M\n");
	return Tm;
}
#undef HISTO_SIZE
#undef HISTO_FRACTION

void increase(int *j, double delta)
{
	int step;
	if((step=(int)((*j)*delta))>0)
		*j+=step;
	else (*j)++;
	return;
}

void shuffle(unsigned long *v, int n)
{
	int i, j;
	int s;

	for (i = n-1; i > 0; i--) {
		j = (int)(((double)rand() / RAND_MAX) * i);
		s = v[j];
		v[j] = v[i];
		v[i] = s;
	}
}

void Legendre_basis(int p, int q)
{
	void normalize(double *v, double norm, int N);
	int i,tw,j,k,l;  
	double c; // normalization constant
	double n,sum1,sum2;
	tw=2*p+1;
	for(i=0,c=0.0;i<tw;i++){
		base[0][i]=1.;
		c+=base[0][i]*base[0][i];
	}
	c=sqrt(c);
	normalize(base[0],c,tw);
	for(j=1;j<q;j++){
		for(i=0,c=0.0;i<tw;i++){
			n=(double)(i-p);
			for(k=0,sum1=0.0;k<j;k++){
				for(l=0,sum2=0.0;l<tw;l++) sum2+=pow(l-p,j)*base[k][l];
				sum1+=base[k][i]*sum2;
			}
			base[j][i]=pow(n,j)-sum1;
			c+=base[j][i]*base[j][i];
		}
		c=sqrt(c);
		normalize(base[j],c,tw);
	}
		
}

void normalize(double *v, double norm, int N)
{
	int i;
	for(i=0;i<N;i++) v[i]/=norm;
}

void embed1(double *x, int m)
{
    int i,j,t;
    for(j=0;j<m;j++)
	   barx[j]=x-delaylist[j];
}

void embed2(double *x, int tw, int m)
{
	int i,j,t;
	double min,max;
	max=-1.0;
	for(i=0;i<m;i++){
		min=1e60;
		for(j=0;j<lmax;j++)
		{
			barx[i][j]=0.0;
			for(t=0;t<tw;t++)
				barx[i][j]+=x[j-t]*base[i][t];
			if(barx[i][j]<min) min=barx[i][j];
			
		}
		for(j=0;j<lmax;j++){
			barx[i][j]-=min;
			if(barx[i][j]>max) max=barx[i][j];
		}
	}
	if (max>0){
		for(i=0;i<m;i++)
		for(j=0;j<lmax;j++)
			barx[i][j]/max;
	}
	
}
double distance(int n, int m)
{
	int i;
	double aux,dx,dxi;
	dx=0.0; 
	for(i=0;i<dim;i++){
		dxi=barx[i][m]-barx[i][n];
		dx+=dxi*dxi;
	}
	return sqrt(dx/dim);
}

void rescale(double **barx,int dim,unsigned long l,int *j1, int *j2)
// output: j1 and j2 (<dim) coordinates wich bigger variance
// shift barx data to positives values
// rescale barx data in order to put j1,j2 cordinates inside the box [0,1][0,1]
{
	int j,i;
	double av,sd1=-1.0,sd2=-1.0,sd;
	double min,max;
	for(j=0;j<Mdim;j++){
		if(j<dim){
			variance(barx[j],l,&av,&sd);
			if(sd>sd1){sd2=sd1;*j2=*j1;sd1=sd;*j1=j;}
			else if(sd>sd2) {sd2=sd;*j2=j;}
		}
		min=barx[j][0];
		for (i=1;i<l;i++) {
			if (barx[j][i] < min) min=barx[j][i];
		}
		for (i=0;i<l;i++) barx[j][i]-=min;
	}
	max=-1.0;
	for (i=0;i<l;i++) {
		if (barx[*j1][i] > max) max=barx[*j1][i];
		if (barx[*j2][i] > max) max=barx[*j2][i];
	}
	if(max>0){
		for(j=0;j<Mdim;j++)
			for (i=0;i<l;i++) barx[j][i]/=max;
	}
	else {
		fprintf(stderr,"rescale: no data range. It makes\n"
		"\t\tno sense to continue. Exiting!\n\n");
		exit(RESCALE_DATA_ZERO_INTERVAL);
	}
}

void mmb(double eps)
// xdim,ydim: the two directions that most spread the attractor
{
	int i;
	int x,y;

	for (x=0;x<BOX;x++){
		for (y=0;y<BOX;y++){
			box[x][y] = -1;
			boxsize[x][y] = 1;
		}
	}

	for (i=0;i<lmax;i++) {
		x=(int)(barx[xdim][i]/eps)&ibox;
		y=(int)(barx[ydim][i]/eps)&ibox;
		list[i]=box[x][y];
		box[x][y]=i;
	}
}

int order(int element, double dx, unsigned long *knn, double *dknn, int m, int M)
// inserta elementos en orden creciente en knn,dknn
// retorna 1 si se incrementó la lista de vecinos
// retorna 0 si la lista de vecinos mantiene la longitud
// no agrega elementos que no cumplan con la theiler window con alguno de los existentes (se queda con el más cercano)
{
	int i=0;
	int auxelem,auxelem2;
	double auxdx,auxdx2;
	int add=0;
	for(i=0;(i<m)&&(dx>=dknn[i]);i++) //busco la posición correcta
		if(labs(element-knn[i]) <= theiler) return 0; //chequeno que no este dentro de la theiler window de ningun elemento
  
	if(i==M) return 0;// si llegue al final salgo sin agregar
	add=1; // si no, agrego un elemento a la lista
	auxelem=knn[i]; knn[i]=element; // intercambio nuevo elemento por el i-ésimo
	auxdx=dknn[i]; dknn[i]=dx;
	i++;
	while((i<m+add)&&(i<M)){ //Desplazo el resto de la lista a la vez que descarto elemento dentro de la theilerW
		while((labs(element-auxelem) <= theiler)&&(i<m+add)){ //si está dentro de la theilerW
			add--; // tengo menos elementos
			if(i==(m+add)) return add; // si llegue al final salgo
			auxelem=knn[i-add]; auxdx=dknn[i-add];//reemplazo auxelem por el siguiente elemento de la serie
		}
		if(i+1-add<m){ auxelem2=knn[i+1-add]; auxdx2=dknn[i+1-add];}
		knn[i]=auxelem; dknn[i]=auxdx;// hago el desplazamiento
		auxelem=auxelem2; auxdx=auxdx2;
		i++;
	}
	if((m==M)&&(add==1)) return 0; // si la lista está llena no puedo agregar más
	return add;
}

int find_nearest(int n, unsigned long *knn, double *dknn, double eps, int M)
{
	int x,y,x1,x2,y1,y2,l,lmax,i,k,kmax;
	int element;
	double dx,aux;
	int m;
	x=(long)(barx[xdim][n]/eps)&ibox;
	y=(long)(barx[ydim][n]/eps)&ibox;
	l=boxsize[x][y];
	m=0;
	lmax=(int)(1./eps)+1;
	while(m<M){
		aux=eps*l;
		dknn[M-1]=aux;
		kmax=1; // intento con una pasada llenar la lista
		for(k=0;((k<kmax)&&(m<M));k++) //para que el algorimo no falle nunca a veces necesito dos pasadas
		for (x1=x-l;x1<=x+l;x1++) {
			x2=x1&ibox;
			for (y1=y-l;y1<=y+l;y1++) {
				element=box[x2][(y2=y1&ibox)];
				while (element != -1) {
					if (labs(element-n) > theiler) {
						dx=distance(n,element);
						if ((dx<dknn[M-1])||(l>lmax)){ // tiene que ser menor al tamaño de caja o al más lejano
							m+=order(element,dx,knn,dknn,m,M); // try to insert the element in the ordered list
							if(m==M) kmax=2; // la lista se llenó al menos una vez.
						}
					}
					element=list[element];
				}
				boxsize[x2][y2]=l; // igualo el tamaño de caja de los vecinos para proximas búsquedas
			}
		}
		l*=2;
	}
	return m;
}

double E2_x(double *x, unsigned long *nlist, int kmax, int T0, int T)
{
    int k,n;
    int t;
    double tmed,tvar,var,aux;
    var=0.0;
    for(t=T0;t<=T;t++)
    {
	 tmed=0.0;tvar=0.0;
	 for(k=0;k<kmax;k++){
		    n=nlist[k];
		    tvar+=(aux=x[n+t])*aux;
		    tmed+=x[n+t];
	 }
	 tvar/=kmax;
	 tmed/=kmax;
	 tvar-=tmed*tmed;
	 var+=tvar;
    }
    return var/((T-T0+1)*variance_data);
}

double epsilon2(unsigned long *nlist, int kmax)
{
    int k1,k2,n1,n2,npar=0;
    double r=0.0,aux;
    for(k1=0;k1<kmax;k1++)
    for(k2=0;k2<k1;k2++){
        n1=nlist[k1];
        n2=nlist[k2];
        r+=(aux=distance(n1,n2))*aux;
	   npar++;
    }
    r/=npar;
    return r;
}

void fill_delay_list(int d, int m)
{
        int i;
        for(i=0;i<m;i++)
                delaylist[i]=i*d;
}
