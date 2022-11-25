import torch

def get_tps_para(B, scal=0.9, scal_var=0.05, tps_scal=0.05, off_scal=0.05, rot_scal=0.5):
    def get_rot_mat(rot_param):
        c = torch.cos(rot_param)
        s = torch.sin(rot_param)
        rot_mat = torch.zeros([rot_param.shape[0],2,2])
        rot_mat[:,0,0] = c
        rot_mat[:,0,1] = -s
        rot_mat[:,1,0] = s
        rot_mat[:,1,1] = c
        return rot_mat

    coord = torch.tensor([[-0.7, -0.7], [0.7, -0.7], [-0.7, 0.7], [0.7, 0.7],
                        [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [-0.5, - 0.5]])
    N,_ = coord.shape
    coord = coord.unsqueeze(0).reshape(1,N,2).repeat(B,1,1)

    coord = coord + (torch.rand(coord.shape)-0.5)/5

    vector = (torch.rand(coord.shape)-0.5)*2*tps_scal

    offset = (torch.rand([B,1,2])-0.5)*2*off_scal

    offset_2 = (torch.rand([B,1,2])-0.5)*2*off_scal

    t_scal = (torch.rand([B,1,2])-0.5)*2*scal_var+scal

    rot_param = (torch.rand([B])-0.5)*2 * rot_scal
    rot_mat = get_rot_mat(rot_param)

    scaled_coord = t_scal*(coord + vector - offset) + offset

    t_vector = torch.matmul(rot_mat, (scaled_coord - offset_2).permute(0,2,1)).permute(0,2,1) + offset_2 - coord
    
    f_vector = torch.FloatTensor([[[-0.99,-0.99],[0,-0.99],[0.99,-0.99],[0.99,0],[0.99,0.99],[0,0.99],[-0.99,0.99],[-0.95,0]]])

    coord = torch.cat((coord,f_vector),dim=1)
    t_vector = torch.cat((t_vector,torch.zeros_like(f_vector)),dim=1)
    return coord, t_vector


def get_thin_plate_spline_grid(coord, vector, out_size):
    def _solve_system(coord, vector):
        B,N,_ = coord.shape    
        ones = torch.ones([B,N,1])

        p = torch.cat([ones,coord],axis=2)
        p_1 = p.reshape(B,-1,1,3)
        p_2 = p.reshape(B,1,-1,3)
        d2 = torch.sum(torch.square(p_1-p_2), axis=3)
        r = d2 * torch.log(d2 + 1e-6)
        zeros = torch.zeros([B, 3, 3])
        W_0 = torch.cat([p, r], 2) 
        W_1 = torch.cat([zeros, p.permute([0, 2, 1])], 2) 
        W = torch.cat([W_0, W_1], 1)
        W_inv = torch.inverse(W)
        tp = torch.cat([coord+vector,torch.zeros(B,3,2)],1)
        T = torch.matmul(W_inv, tp).permute([0,2,1])
        return T
    def _meshgrid(height, width, coord):
        B,N,_ = coord.shape

        x_t_flat = torch.linspace(-1,1,width).repeat(height).reshape(1,1,-1)
        y_t_flat = torch.linspace(-1,1,height).repeat_interleave(width).reshape(1,1,-1)

        px = coord[:,:,[0]]
        py = coord[:,:,[1]]
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)

        r = d2*torch.log(d2+1e-6)

        x_t_flat_g = x_t_flat.repeat(B,1,1)
        y_t_flat_g = y_t_flat.repeat(B,1,1)
        ones = torch.ones(x_t_flat_g.shape)

        grid = torch.cat([ones,x_t_flat_g,y_t_flat_g,r],axis=1)
        return grid
    def gen_grid(T, coord, out_size):
        B,_,_ = T.shape
        H_out = out_size[0]
        W_out = out_size[1]

        grid = _meshgrid(H_out, W_out, coord)

        T_g = torch.matmul(T, grid)
    
        # input_transformed = torch.nn.functional.grid_sample(input_dim, T_g.reshape(B,2,H_out,W_out).permute(0,2,3,1))
        return T_g.reshape(B,2,H_out,W_out).permute(0,2,3,1)
    T = _solve_system(coord, vector)
    T_g = gen_grid(T,coord,out_size)
    # image = _transform(T, coord, U, out_size)
    return T_g

